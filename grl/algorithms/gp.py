import os
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import d4rl
import wandb
from grl.agents.gp import GPAgent

from grl.datasets import create_dataset
from grl.datasets.gpo import GPDataset, GPD4RLDataset
from grl.generative_models.diffusion_model import DiffusionModel
from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.generative_models.bridge_flow_model.schrodinger_bridge_conditional_flow_model import (
    SchrodingerBridgeConditionalFlowModel,
)
from grl.generative_models.bridge_flow_model.guided_bridge_conditional_flow_model import (
    GuidedBridgeConditionalFlowModel,
)
from grl.generative_models.diffusion_model.guided_diffusion_model import (
    GuidedDiffusionModel,
)
from grl.generative_models.conditional_flow_model.guided_conditional_flow_model import (
    GuidedConditionalFlowModel,
)
from grl.neural_network import MultiLayerPerceptron
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.utils import set_seed
from grl.utils.statistics import sort_files_by_criteria
from grl.generative_models.metric import compute_likelihood


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MultiLayerPerceptron(
            hidden_sizes=[state_dim, 256, 256],
            output_size=1,
            activation="relu",
        )

    def forward(self, state):
        return self.v(state)


class GPCritic(nn.Module):
    """
    Overview:
        Critic network for GP algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of GP critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.q_alpha = config.q_alpha
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of GP critic.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        return self.q(action, state)

    def compute_double_q(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Return the output of two Q networks.
        Arguments:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            q1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first Q network.
            q2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second Q network.
        """
        return self.q.compute_double_q(action, state)

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
        discount_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        """
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = (
                self.q_target(
                    fake_next_action,
                    torch.stack([next_state] * fake_next_action.shape[1], axis=1),
                )
                .detach()
                .squeeze(dim=-1)
            )
            next_v = torch.sum(
                softmax(self.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True
            )
        # Update Q function
        targets = reward + (1.0 - done.float()) * discount_factor * next_v.detach()
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (
            torch.nn.functional.mse_loss(q0, targets)
            + torch.nn.functional.mse_loss(q1, targets)
        ) / 2
        return q_loss, torch.mean(q0), torch.mean(targets)

    def v_loss(self, vf, data, tau):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        with torch.no_grad():
            target_q = self.q_target(a, s).detach()
            next_v = vf(s_).detach()
        # Update value function
        v = vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, tau)
        return v_loss, next_v

    def iql_q_loss(self, data, next_v, discount):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        d = data["d"]
        q_target = r + (1.0 - d.float()) * discount * next_v.detach()
        qs = self.q.compute_double_q(a, s)
        q_loss = sum(torch.nn.functional.mse_loss(q, q_target) for q in qs) / len(qs)
        return q_loss, torch.mean(qs[0]), torch.mean(q_target)


class GuidedPolicy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.type = config.model_type
        if self.type == "DiffusionModel":
            self.model = GuidedDiffusionModel(config.model)
        elif self.type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
        ]:
            self.model = GuidedConditionalFlowModel(config.model)
        elif self.type in ["SchrodingerBridgeConditionalFlowModel"]:
            self.model = GuidedBridgeConditionalFlowModel(config.model)
        else:
            raise NotImplementedError

    def sample(
        self,
        base_model,
        guided_model,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        guidance_scale: Union[torch.Tensor, float] = 1.0,
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of GP policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """

        if self.type == "DiffusionModel":
            if guidance_scale == 0.0:
                return base_model.sample(
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    with_grad=False,
                    solver_config=solver_config,
                )
            elif guidance_scale == 1.0:
                return guided_model.sample(
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    with_grad=False,
                    solver_config=solver_config,
                )
            else:
                return self.model.sample(
                    base_model=base_model,
                    guided_model=guided_model,
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    guidance_scale=guidance_scale,
                    with_grad=False,
                    solver_config=solver_config,
                )

        elif self.type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:

            x_0 = base_model.gaussian_generator(batch_size=state.shape[0])

            if guidance_scale == 0.0:
                return base_model.sample(
                    x_0=x_0,
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    with_grad=False,
                    solver_config=solver_config,
                )
            elif guidance_scale == 1.0:
                return guided_model.sample(
                    x_0=x_0,
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    with_grad=False,
                    solver_config=solver_config,
                )
            else:
                return self.model.sample(
                    base_model=base_model,
                    guided_model=guided_model,
                    x_0=x_0,
                    t_span=t_span,
                    condition=state,
                    batch_size=batch_size,
                    guidance_scale=guidance_scale,
                    with_grad=False,
                    solver_config=solver_config,
                )


class GPPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = GPCritic(config.critic)
        self.model_type = config.model_type
        if self.model_type == "DiffusionModel":
            self.base_model = DiffusionModel(config.model)
            self.guided_model = DiffusionModel(config.model)
            self.model_loss_type = config.model_loss_type
            assert self.model_loss_type in ["score_matching", "flow_matching"]
        elif self.model_type == "OptimalTransportConditionalFlowModel":
            self.base_model = OptimalTransportConditionalFlowModel(config.model)
            self.guided_model = OptimalTransportConditionalFlowModel(config.model)
        elif self.model_type == "IndependentConditionalFlowModel":
            self.base_model = IndependentConditionalFlowModel(config.model)
            self.guided_model = IndependentConditionalFlowModel(config.model)
        elif self.model_type == "SchrodingerBridgeConditionalFlowModel":
            self.base_model = SchrodingerBridgeConditionalFlowModel(config.model)
            self.guided_model = SchrodingerBridgeConditionalFlowModel(config.model)
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, state: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of GP policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.sample(state)

    def sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
        with_grad: bool = False,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of GP policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """

        return self.guided_model.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            with_grad=with_grad,
            solver_config=solver_config,
        )

    def behaviour_policy_sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
        with_grad: bool = False,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behaviour policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
            with_grad (:obj:`bool`): Whether to calculate the gradient.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.base_model.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            with_grad=with_grad,
            solver_config=solver_config,
        )

    def compute_q(
        self,
        state: Union[torch.Tensor, TensorDict],
        action: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q value.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
        Returns:
            q (:obj:`torch.Tensor`): The Q value.
        """

        return self.critic(action, state)

    def behaviour_policy_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if self.model_type == "DiffusionModel":
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    return self.base_model.score_matching_loss(action, state)
                else:
                    return self.base_model.score_matching_loss(
                        action, state, weighting_scheme="vanilla"
                    )
            elif self.model_loss_type == "flow_matching":
                return self.base_model.flow_matching_loss(action, state)
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.base_model.gaussian_generator(batch_size=state.shape[0])
            return self.base_model.flow_matching_loss(x0=x0, x1=action, condition=state)

    def policy_loss_withgrade(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
        loss_type: str = "origin_loss",
        gradtime_step: int = 1000,
        eta: float = 1.0,
        repeats: int = 10,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if self.model_type == "DiffusionModel":
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, average=True
                    )
                else:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, weighting_scheme="vanilla", average=True
                    )
            elif self.model_loss_type == "flow_matching":
                model_loss = self.guided_model.flow_matching_loss(
                    action, state, average=True
                )
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.guided_model.gaussian_generator(batch_size=state.shape[0])
            model_loss = self.guided_model.flow_matching_loss(
                x0=x0, x1=action, condition=state, average=True
            )
        else:
            raise NotImplementedError
        t_span = torch.linspace(0.0, 1.0, gradtime_step).to(state.device)

        state_repeated = torch.repeat_interleave(state, repeats=repeats, dim=0)
        new_action = self.guided_model.sample(
            t_span=t_span, condition=state_repeated, with_grad=True
        )
        state_split = state_repeated.view(-1, repeats, *state.size()[1:])
        action_split = new_action.view(-1, repeats, *new_action.size()[1:])
        q_values = []
        for i in range(repeats):
            state_i = state_split[:, i, :]
            action_i = action_split[:, i, :]
            q_value_i = self.critic(
                action_i,
                state_i,
            )
            q_values.append(q_value_i)
        q_values_stack = torch.stack(q_values, dim=0)
        q_values_sum = q_values_stack.sum(dim=0)
        q_value = q_values_sum / repeats
        if loss_type == "origin_loss":
            return -q_value.mean() + model_loss
        elif loss_type == "detach_loss":
            return -(q_value / q_value.abs().detach()).mean() * eta + model_loss
        elif loss_type == "minibatch_loss":
            q_loss = -q_value.mean() / q_value.abs().mean().detach()
            return eta * q_loss + model_loss
        elif loss_type == "double_minibatch_loss":
            q_loss = -q_value.mean() / q_value.abs().mean().detach()
            return eta * q_loss + model_loss
            # q1, q2 = self.critic.q.compute_double_q(new_action, state)
            # if np.random.uniform() > 0.5:
            #     q_loss = -q1.mean() / q2.abs().mean().detach()
            # else:
            #     q_loss = -q2.mean() / q1.abs().mean().detach()
            # return eta * q_loss + model_loss
        else:
            raise ValueError(("Unknown activation function {}".format(loss_type)))

    def policy_loss_pure_grad(
        self,
        state: Union[torch.Tensor, TensorDict],
        loss_type: str = "origin_loss",
        gradtime_step: int = 1000,
        eta: float = 1.0,
        repeats: int = 1,
        value_function: ValueFunction = None,
    ):
        t_span = torch.linspace(0.0, 1.0, gradtime_step).to(state.device)

        def log_grad(name, grad):
            wandb.log(
                {
                    f"{name}_mean": grad.mean().item(),
                    f"{name}_max": grad.max().item(),
                    f"{name}_min": grad.min().item(),
                },
                commit=False,
            )

        state_repeated = torch.repeat_interleave(
            state, repeats=repeats, dim=0
        ).requires_grad_()
        state_repeated.register_hook(lambda grad: log_grad("state_repeated", grad))
        action_repeated = self.guided_model.sample(
            t_span=t_span, condition=state_repeated, with_grad=True
        )
        action_repeated.register_hook(lambda grad: log_grad("action_repeated", grad))

        q_value_repeated = self.critic(action_repeated, state_repeated)
        q_value_repeated.register_hook(lambda grad: log_grad("q_value_repeated", grad))
        log_p = compute_likelihood(
            model=self.guided_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        log_p.register_hook(lambda grad: log_grad("log_p", grad))

        bits_ratio = torch.prod(
            torch.tensor(state_repeated.shape[1], device=state.device)
        ) * torch.log(torch.tensor(2.0, device=state.device))
        log_p_mean_per_dim = log_p.mean() / bits_ratio
        log_mu = compute_likelihood(
            model=self.base_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        log_mu.register_hook(lambda grad: log_grad("log_mu", grad))
        log_mu_mean_per_dim = log_mu.mean() / bits_ratio
        if loss_type == "origin_loss":
            return (
                -q_value_repeated.mean()
                + (log_p_mean_per_dim - log_mu_mean_per_dim) / eta
            )
        elif loss_type == "detach_loss":
            return (
                -(q_value_repeated / q_value_repeated.abs().detach()).mean()
                + (log_p_mean_per_dim - log_mu_mean_per_dim) / eta
            )
        elif loss_type == "vf_loss":
            v_value = value_function(state_repeated).squeeze(dim=-1)
            return (
                -(q_value_repeated / v_value.abs().detach()).mean()
                + (log_p_mean_per_dim - log_mu_mean_per_dim) / eta
            )

    def policy_loss_pure_grad_polish(
        self,
        state: Union[torch.Tensor, TensorDict],
        loss_type: str = "origin_loss",
        gradtime_step: int = 1000,
        eta: float = 1.0,
        repeats: int = 1,
        value_function: ValueFunction = None,
        weight_clamp: float = 100.0,
    ):
        t_span = torch.linspace(0.0, 1.0, gradtime_step).to(state.device)

        state_repeated = torch.repeat_interleave(state, repeats=repeats, dim=0)
        action_repeated = self.base_model.sample(
            t_span=t_span, condition=state_repeated, with_grad=False
        )
        q_value_repeated = self.critic(action_repeated, state_repeated).squeeze(dim=-1)
        v_value_repeated = value_function(state_repeated).squeeze(dim=-1)

        weight = (
            torch.exp(eta * (q_value_repeated - v_value_repeated)).clamp(
                max=weight_clamp
            )
            / weight_clamp
        )

        log_p = compute_likelihood(
            model=self.guided_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        bits_ratio = torch.prod(
            torch.tensor(state_repeated.shape[1], device=state.device)
        ) * torch.log(torch.tensor(2.0, device=state.device))
        log_p_per_dim = log_p / bits_ratio
        log_mu = compute_likelihood(
            model=self.base_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        log_mu_per_dim = log_mu / bits_ratio

        loss = (
            (
                -eta * q_value_repeated.detach()
                + log_p_per_dim.detach()
                - log_mu_per_dim.detach()
            )
            * log_p_per_dim
            * weight
        )
        loss_q = (-eta * q_value_repeated.detach()).mean()
        loss_p = (log_p_per_dim.detach() * log_p_per_dim * weight).mean()
        loss_u = (-log_mu_per_dim.detach() * log_p_per_dim * weight).mean()
        return loss.mean(), loss_q, loss_p, loss_u

    def policy_loss_pure_grad_softmax(
        self,
        state: Union[torch.Tensor, TensorDict],
        gradtime_step: int = 1000,
        eta: float = 1.0,
        repeats: int = 10,
    ):
        assert repeats > 1
        t_span = torch.linspace(0.0, 1.0, gradtime_step).to(state.device)

        state_repeated = torch.repeat_interleave(state, repeats=repeats, dim=0)
        action_repeated = self.base_model.sample(
            t_span=t_span, condition=state_repeated, with_grad=False
        )
        q_value_repeated = self.critic(action_repeated, state_repeated).squeeze(dim=-1)
        q_value_reshaped = q_value_repeated.reshape(-1, repeats)

        weight = nn.Softmax(dim=1)(q_value_reshaped * eta)
        weight = weight.reshape(-1)

        log_p = compute_likelihood(
            model=self.guided_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        bits_ratio = torch.prod(
            torch.tensor(state_repeated.shape[1], device=state.device)
        ) * torch.log(torch.tensor(2.0, device=state.device))
        log_p_per_dim = log_p / bits_ratio
        log_mu = compute_likelihood(
            model=self.base_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        log_mu_per_dim = log_mu / bits_ratio

        loss = (
            (
                -eta * q_value_repeated.detach()
                + log_p_per_dim.detach()
                - log_mu_per_dim.detach()
            )
            * log_p_per_dim
            * weight
        )
        loss_q = (-eta * q_value_repeated.detach()).mean()
        loss_p = (log_p_per_dim.detach() * log_p_per_dim * weight).mean()
        loss_u = (-log_mu_per_dim.detach() * log_p_per_dim * weight).mean()
        return loss.mean(), loss_q, loss_p, loss_u

    def policy_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        fake_action: Union[torch.Tensor, TensorDict] = None,
        maximum_likelihood: bool = False,
        eta: float = 1.0,
        regularize_method: str = "minus_value",
        value_function: ValueFunction = None,
        weight_clamp: float = 100.0,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        if self.model_type == "DiffusionModel":
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, average=False
                    )
                else:
                    model_loss = self.guided_model.score_matching_loss(
                        action, state, weighting_scheme="vanilla", average=False
                    )
            elif self.model_loss_type == "flow_matching":
                model_loss = self.guided_model.flow_matching_loss(
                    action, state, average=False
                )
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.guided_model.gaussian_generator(batch_size=state.shape[0])
            model_loss = self.guided_model.flow_matching_loss(
                x0=x0, x1=action, condition=state, average=False
            )
        else:
            raise NotImplementedError

        if regularize_method == "origin":

            with torch.no_grad():
                q_value = self.critic(action, state).squeeze(dim=-1)

                weight = torch.exp(eta * q_value)

        elif regularize_method == "minus_value":

            with torch.no_grad():
                q_value = self.critic(action, state).squeeze(dim=-1)

                if value_function is not None:
                    v_value = value_function(state).squeeze(dim=-1)

                else:

                    fake_q_value = (
                        self.critic(
                            fake_action,
                            torch.stack([state] * fake_action.shape[1], axis=1),
                        )
                        .squeeze(dim=-1)
                        .detach()
                        .squeeze(dim=-1)
                    )

                    v_value = torch.sum(
                        self.softmax(self.critic.q_alpha * fake_q_value) * fake_q_value,
                        dim=-1,
                        keepdim=True,
                    ).squeeze(dim=-1)

                weight = torch.exp(eta * (q_value - v_value))

        clamped_weight = weight.clamp(max=weight_clamp)

        # calculate the number of clamped_weight<weight
        clamped_ratio = torch.mean(
            torch.tensor(clamped_weight < weight, dtype=torch.float32)
        )

        return (
            torch.mean(model_loss * clamped_weight),
            torch.mean(weight),
            torch.mean(clamped_weight),
            clamped_ratio,
        )

    def policy_loss_softmax(
        self,
        # action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        fake_action: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
        eta: float = 1.0,
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """
        action = fake_action

        action_reshape = action.reshape(
            action.shape[0] * action.shape[1], *action.shape[2:]
        )
        state_repeat = torch.stack([state] * action.shape[1], axis=1)
        state_repeat_reshape = state_repeat.reshape(
            state_repeat.shape[0] * state_repeat.shape[1], *state_repeat.shape[2:]
        )
        energy = self.critic(action_reshape, state_repeat_reshape).detach()
        energy = energy.reshape(action.shape[0], action.shape[1]).squeeze(dim=-1)

        if self.model_type == "DiffusionModel":
            if self.model_loss_type == "score_matching":
                if maximum_likelihood:
                    model_loss = self.guided_model.score_matching_loss(
                        action_reshape, state_repeat_reshape, average=False
                    )
                else:
                    model_loss = self.guided_model.score_matching_loss(
                        action_reshape,
                        state_repeat_reshape,
                        weighting_scheme="vanilla",
                        average=False,
                    )
            elif self.model_loss_type == "flow_matching":
                model_loss = self.guided_model.flow_matching_loss(
                    action_reshape, state_repeat_reshape, average=False
                )
        elif self.model_type in [
            "OptimalTransportConditionalFlowModel",
            "IndependentConditionalFlowModel",
            "SchrodingerBridgeConditionalFlowModel",
        ]:
            x0 = self.guided_model.gaussian_generator(
                batch_size=state.shape[0] * action.shape[1]
            )
            model_loss = self.guided_model.flow_matching_loss(
                x0=x0, x1=action_reshape, condition=state_repeat_reshape, average=False
            )
        else:
            raise NotImplementedError

        model_loss = model_loss.reshape(action.shape[0], action.shape[1]).squeeze(
            dim=-1
        )

        relative_energy = nn.Softmax(dim=1)(energy * eta)

        loss = torch.mean(torch.sum(relative_energy * model_loss, axis=-1))

        return (
            loss,
            torch.mean(energy),
            torch.mean(relative_energy),
            torch.mean(model_loss),
        )

    def q_loss(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        reward: Union[torch.Tensor, TensorDict],
        next_state: Union[torch.Tensor, TensorDict],
        done: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
        discount_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the Q loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        """
        return self.critic.q_loss(
            action, state, reward, next_state, done, fake_next_action, discount_factor
        )


class GPAlgorithm:

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: GPDataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
        seed=None,
    ):
        """
        Overview:
            Initialize the GP algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            dataset (:obj:`GPDataset`): The dataset.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator
        self.dataset = dataset
        self.seed_value = set_seed()  # set_seed(seed_value=seed)

        # ---------------------------------------
        # Customized model initialization code ↓
        # ---------------------------------------

        if model is not None:
            self.model = model
            self.behaviour_policy_train_epoch = 0
            self.critic_train_epoch = 0
            self.guided_policy_train_epoch = 0
        else:
            self.model = torch.nn.ModuleDict()
            config = self.config.train
            assert hasattr(config.model, "GPPolicy")

            if torch.__version__ >= "2.0.0":
                self.model["GPPolicy"] = torch.compile(
                    GPPolicy(config.model.GPPolicy).to(config.model.GPPolicy.device)
                )
            else:
                self.model["GPPolicy"] = GPPolicy(config.model.GPPolicy).to(
                    config.model.GPPolicy.device
                )
            self.model["GuidedPolicy"] = GuidedPolicy(config=config.model.GuidedPolicy)

            if config.parameter.critic.method == "iql":
                self.vf = ValueFunction(
                    config.model.GPPolicy.model.model.args.backbone.args.condition_dim
                ).to(config.model.GPPolicy.device)

            if (
                hasattr(config.parameter, "checkpoint_transform")
                and config.parameter.checkpoint_transform == True
            ):
                if (
                    hasattr(config.parameter, "checkpoint_path")
                    and config.parameter.checkpoint_path is not None
                ):

                    if not os.path.exists(config.parameter.checkpoint_path):
                        log.warning(
                            f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                        )
                        self.behaviour_policy_train_epoch = 0
                        self.critic_train_epoch = 0
                        self.guided_policy_train_epoch = 0
                    else:
                        checkpoint_files = sort_files_by_criteria(
                            folder_path=config.parameter.checkpoint_path,
                            start_string="checkpoint_",
                            end_string=".pt",
                        )
                        value_function_files = sort_files_by_criteria(
                            folder_path=config.parameter.checkpoint_path,
                            start_string="valuefunction_",
                            end_string=".pt",
                        )
                        if not config.parameter.critic.method == "iql":
                            log.info("we don't use iql for critic training")
                            critic_train_epoch_1 = 0
                        elif len(value_function_files) == 0:
                            critic_train_epoch_1 = 0
                        else:
                            checkpoint = torch.load(
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    value_function_files[0],
                                ),
                                map_location="cpu",
                            )
                            self.vf.load_state_dict(checkpoint["model"])
                            critic_train_epoch_1 = checkpoint.get(
                                "critic_train_epoch", 0
                            )

                        if len(checkpoint_files) == 0:
                            self.behaviour_policy_train_epoch = 0
                            self.critic_train_epoch = 0
                            self.guided_policy_train_epoch = 0
                            log.warning(
                                f"No checkpoint file found in {config.parameter.checkpoint_path}"
                            )
                        else:
                            checkpoint = torch.load(
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    checkpoint_files[0],
                                ),
                                map_location="cpu",
                            )

                            from collections import OrderedDict

                            checkpoint_sorted = OrderedDict()
                            checkpoint_sorted_2 = OrderedDict()
                            for key, value in checkpoint["model"].items():
                                name = key.replace("module.", "")
                                checkpoint_sorted_2[name] = value
                            for key, value in checkpoint_sorted_2.items():
                                name = key.replace("GPOPolicy.", "GPPolicy.")
                                checkpoint_sorted[name] = value
                            self.model.load_state_dict(checkpoint_sorted)
                            self.behaviour_policy_train_epoch = checkpoint.get(
                                "behaviour_policy_train_epoch", 0
                            )
                            self.critic_train_epoch = checkpoint.get(
                                "critic_train_epoch", 0
                            )
                            self.guided_policy_train_epoch = checkpoint.get(
                                "guided_policy_train_epoch", 0
                            )
                            assert critic_train_epoch_1 == self.critic_train_epoch
                            log.info(
                                f"Load checkpoint: behaviour_policy_train_epoch: {self.behaviour_policy_train_epoch}, critic_train_epoch: {self.critic_train_epoch}, guided_policy_train_epoch: {self.guided_policy_train_epoch}"
                            )
                else:
                    self.behaviour_policy_train_epoch = 0
                    self.critic_train_epoch = 0
                    self.guided_policy_train_epoch = 0
                torch.save(
                    dict(
                        base_model=self.model["GPPolicy"].base_model.state_dict(),
                        behaviour_policy_train_epoch=self.behaviour_policy_train_epoch,
                    ),
                    f=os.path.join(
                        config.parameter.checkpoint_path,
                        f"basemodel_{self.behaviour_policy_train_epoch}.pt",
                    ),
                )
                torch.save(
                    dict(
                        guided_model=self.model["GPPolicy"].guided_model.state_dict(),
                        guided_policy_train_epoch=self.guided_policy_train_epoch,
                    ),
                    f=os.path.join(
                        config.parameter.checkpoint_path,
                        f"guidedmodel_{self.guided_policy_train_epoch}_{0}.pt",
                    ),
                )
                torch.save(
                    dict(
                        critic_model=self.model["GPPolicy"].critic.state_dict(),
                        critic_train_epoch=self.critic_train_epoch,
                        value_function=self.vf.state_dict(),
                    ),
                    f=os.path.join(
                        config.parameter.checkpoint_path,
                        f"critic_{self.critic_train_epoch}.pt",
                    ),
                )
            if (
                hasattr(config.parameter, "checkpoint_path")
                and config.parameter.checkpoint_path is not None
            ):
                if not os.path.exists(config.parameter.checkpoint_path):
                    log.warning(
                        f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                    )
                    self.behaviour_policy_train_epoch = 0
                    self.critic_train_epoch = 0
                    self.guided_policy_train_epoch = 0
                else:
                    for filename in os.listdir(config.parameter.checkpoint_path):
                        if "__" in filename:
                            new_filename = filename.replace("__", "_")
                            old_file = os.path.join(
                                config.parameter.checkpoint_path, filename
                            )
                            new_file = os.path.join(
                                config.parameter.checkpoint_path, new_filename
                            )
                            os.rename(old_file, new_file)
                    base_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="basemodel_",
                        end_string=".pt",
                    )
                    if len(base_model_files) == 0:
                        self.behaviour_policy_train_epoch = 0
                        log.warning(
                            f"No basemodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                base_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model["GPPolicy"].base_model.load_state_dict(
                            checkpoint["base_model"]
                        )
                        self.behaviour_policy_train_epoch = checkpoint.get(
                            "behaviour_policy_train_epoch", 0
                        )

                    guided_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="guidedmodel_",
                        end_string=".pt",
                    )
                    if len(guided_model_files) == 0:
                        self.guided_policy_train_epoch = 0
                        log.warning(
                            f"No guidedmodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                guided_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model["GPPolicy"].guided_model.load_state_dict(
                            checkpoint["guided_model"]
                        )
                        self.guided_policy_train_epoch = checkpoint.get(
                            "guided_policy_train_epoch", 0
                        )

                    critic_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="critic_",
                        end_string=".pt",
                    )
                    if len(critic_model_files) == 0:
                        self.critic_train_epoch = 0
                        log.warning(
                            f"No criticmodel file found in {config.parameter.checkpoint_path}"
                        )
                    else:
                        checkpoint = torch.load(
                            os.path.join(
                                config.parameter.checkpoint_path,
                                critic_model_files[0],
                            ),
                            map_location="cpu",
                        )
                        self.model["GPPolicy"].critic.load_state_dict(
                            checkpoint["critic_model"]
                        )
                        self.critic_train_epoch = checkpoint.get(
                            "critic_train_epoch", 0
                        )
                        if hasattr(self, "vf"):
                            self.vf.load_state_dict(checkpoint["value_function"])

        # ---------------------------------------
        # Customized model initialization code ↑
        # ---------------------------------------

    def train(self, config: EasyDict = None, seed=None):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
            seed (:obj:`int`): The random seed.
        """

        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )

        config["seed"] = self.seed_value

        if not hasattr(config, "wandb"):
            config["wandb"] = dict(project=config.project)
        elif not hasattr(config.wandb, "project"):
            config.wandb["project"] = config.project

        with wandb.init(**config.wandb) as wandb_run:
            if not hasattr(config.parameter.guided_policy, "eta"):
                config.parameter.guided_policy.eta = 1.0
            if not config.model.GPPolicy.model_type == "DiffusionModel":
                path_type = "ICFM"
            else:
                path_type = config.model.GPPolicy.model.path.type
            if not hasattr(config.model.GPPolicy, "model_loss_type"):
                config.model.GPPolicy["model_loss_type"] = "flow_matching"
            if config.parameter.algorithm_type == "GPO":
                run_name = f"Q-{config.parameter.critic.method}-path-{path_type}-eta-{config.parameter.guided_policy.eta}-batch-{config.parameter.guided_policy.batch_size}-lr-{config.parameter.guided_policy.learning_rate}-{config.model.GPPolicy.model.model.type}-{self.seed_value}"
                wandb.run.name = run_name
                wandb.run.save()

            elif config.parameter.algorithm_type in [
                "GPG",
                "GPG_Direct",
                "GPG_Polish",
                "GPG_Softmax",
            ]:
                run_name = f"Q-{config.parameter.critic.method}-path-{path_type}-eta-{config.parameter.guided_policy.eta}-T-{config.parameter.guided_policy.gradtime_step}-batch-{config.parameter.guided_policy.batch_size}-lr-{config.parameter.guided_policy.learning_rate}-seed-{self.seed_value}"
                wandb.run.name = run_name
                wandb.run.save()

            config = merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = (
                create_simulator(config.simulator)
                if hasattr(config, "simulator")
                else self.simulator
            )
            self.dataset = (
                create_dataset(config.dataset)
                if hasattr(config, "dataset")
                else self.dataset
            )

            fake_dataset_path = os.path.join(
                config.parameter.checkpoint_path,
                f"dataset_with_fakeaction.npz",
            )

            if os.path.exists(fake_dataset_path):
                fake_dataset = np.load(fake_dataset_path)

                self.dataset.fake_actions = torch.from_numpy(
                    fake_dataset["fake_actions"]
                ).to(self.dataset.device)
                log.info(
                    f"Test fake actions: {self.dataset.fake_actions[0].cpu().numpy().mean()}"
                )
                self.dataset.fake_next_actions = torch.from_numpy(
                    fake_dataset["fake_next_actions"]
                ).to(self.dataset.device)
                log.info(
                    f"Test fake next actions: {self.dataset.fake_next_actions[0].cpu().numpy().mean()}"
                )
                self.need_fake_action = False
            else:
                self.need_fake_action = True

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------

            def save_checkpoint(model, iteration=None, model_type=False):
                if iteration == None:
                    iteration = 0
                if model_type == "base_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        torch.save(
                            dict(
                                base_model=model["GPPolicy"].base_model.state_dict(),
                                behaviour_policy_train_epoch=self.behaviour_policy_train_epoch,
                                behaviour_policy_train_iter=iteration,
                            ),
                            f=os.path.join(
                                config.parameter.checkpoint_path,
                                f"basemodel_{self.behaviour_policy_train_epoch}_{iteration}.pt",
                            ),
                        )
                elif model_type == "guided_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        torch.save(
                            dict(
                                guided_model=model[
                                    "GPPolicy"
                                ].guided_model.state_dict(),
                                guided_policy_train_epoch=self.guided_policy_train_epoch,
                                guided_policy_train_iteration=iteration,
                            ),
                            f=os.path.join(
                                config.parameter.checkpoint_path,
                                f"guidedmodel_{self.guided_policy_train_epoch}_{iteration}.pt",
                            ),
                        )
                elif model_type == "critic_model":
                    if (
                        hasattr(config.parameter, "checkpoint_path")
                        and config.parameter.checkpoint_path is not None
                    ):
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        if config.parameter.critic.method == "iql":
                            torch.save(
                                dict(
                                    critic_model=model["GPPolicy"].critic.state_dict(),
                                    critic_train_epoch=self.critic_train_epoch,
                                    critic_train_iter=iteration,
                                    value_function=self.vf.state_dict(),
                                ),
                                f=os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"critic_{self.critic_train_epoch}_{iteration}.pt",
                                ),
                            )
                        elif config.parameter.critic.method == "in_support_ql":
                            torch.save(
                                dict(
                                    critic_model=model["GPPolicy"].critic.state_dict(),
                                    critic_train_epoch=self.critic_train_epoch,
                                    critic_train_iter=iteration,
                                ),
                                f=os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"critic_{self.critic_train_epoch}_{iteration}.pt",
                                ),
                            )

            def generate_fake_action(model, states, sample_per_state):

                fake_actions_sampled = []
                for states in track(
                    np.array_split(states, states.shape[0] // 4096 + 1),
                    description="Generate fake actions",
                ):

                    fake_actions_ = model.behaviour_policy_sample(
                        state=states,
                        batch_size=sample_per_state,
                        t_span=(
                            torch.linspace(0.0, 1.0, config.parameter.t_span).to(
                                states.device
                            )
                            if hasattr(config.parameter, "t_span")
                            and config.parameter.t_span is not None
                            else None
                        ),
                    )
                    fake_actions_sampled.append(torch.einsum("nbd->bnd", fake_actions_))

                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, train_epoch, guidance_scales, repeat=1):
                evaluation_results = dict()
                for guidance_scale in guidance_scales:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.GPPolicy.device,
                        ).unsqueeze(0)
                        action = (
                            model["GuidedPolicy"]
                            .sample(
                                base_model=model["GPPolicy"].base_model,
                                guided_model=model["GPPolicy"].guided_model,
                                state=obs,
                                guidance_scale=guidance_scale,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.t_span
                                    ).to(config.model.GPPolicy.device)
                                    if hasattr(config.parameter, "t_span")
                                    and config.parameter.t_span is not None
                                    else None
                                ),
                            )
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        return action

                    eval_results = self.simulator.evaluate(
                        policy=policy, num_episodes=repeat
                    )
                    return_results = [
                        eval_results[i]["total_return"] for i in range(repeat)
                    ]
                    log.info(f"Return: {return_results}")
                    return_mean = np.mean(return_results)
                    return_std = np.std(return_results)
                    return_max = np.max(return_results)
                    return_min = np.min(return_results)
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_mean"
                    ] = return_mean
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_std"
                    ] = return_std
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_max"
                    ] = return_max
                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/return_min"
                    ] = return_min

                    if isinstance(self.dataset, GPD4RLDataset):
                        env_id = config.dataset.args.env_id
                        evaluation_results[
                            f"evaluation/guidance_scale:[{guidance_scale}]/return_mean_normalized"
                        ] = d4rl.get_normalized_score(env_id, return_mean)
                        evaluation_results[
                            f"evaluation/guidance_scale:[{guidance_scale}]/return_std_normalized"
                        ] = d4rl.get_normalized_score(env_id, return_std)
                        evaluation_results[
                            f"evaluation/guidance_scale:[{guidance_scale}]/return_max_normalized"
                        ] = d4rl.get_normalized_score(env_id, return_max)
                        evaluation_results[
                            f"evaluation/guidance_scale:[{guidance_scale}]/return_min_normalized"
                        ] = d4rl.get_normalized_score(env_id, return_min)

                    if repeat > 1:
                        log.info(
                            f"Train epoch: {train_epoch}, guidance_scale: {guidance_scale}, return_mean: {return_mean}, return_std: {return_std}, return_max: {return_max}, return_min: {return_min}"
                        )
                    else:
                        log.info(
                            f"Train epoch: {train_epoch}, guidance_scale: {guidance_scale}, return: {return_mean}"
                        )

                return evaluation_results

            # ---------------------------------------
            # behavior training code ↓
            # ---------------------------------------

            behaviour_policy_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].base_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )
            if (
                hasattr(config.parameter.behaviour_policy, "lr_decy")
                and config.parameter.behaviour_policy.lr_decy is True
            ):

                behaviour_lr_scheduler = CosineAnnealingLR(
                    behaviour_policy_optimizer,
                    T_max=config.parameter.behaviour_policy.epochs,
                    eta_min=0.0,
                )

            behaviour_policy_train_iter = 0
            for epoch in track(
                range(config.parameter.behaviour_policy.epochs),
                description="Behaviour policy training",
            ):
                if self.behaviour_policy_train_epoch >= epoch:
                    if (
                        hasattr(config.parameter.behaviour_policy, "lr_decy")
                        and config.parameter.behaviour_policy.lr_decy is True
                    ):
                        behaviour_lr_scheduler.step()
                    continue

                sampler = torch.utils.data.RandomSampler(
                    self.dataset, replacement=False
                )
                data_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    pin_memory=False,
                    drop_last=True,
                )

                if (
                    hasattr(config.parameter.evaluation, "eval")
                    and config.parameter.evaluation.eval
                ):
                    if (
                        epoch
                        % config.parameter.evaluation.evaluation_behavior_policy_interval
                        == 0
                        or (epoch + 1) == config.parameter.behaviour_policy.epochs
                    ):
                        evaluation_results = evaluate(
                            self.model,
                            train_epoch=epoch,
                            guidance_scales=[0.0],
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )
                        wandb.log(data=evaluation_results, commit=False)

                counter = 1
                behaviour_policy_loss_sum = 0
                for data in data_loader:

                    behaviour_policy_loss = self.model[
                        "GPPolicy"
                    ].behaviour_policy_loss(
                        data["a"],
                        data["s"],
                        maximum_likelihood=(
                            config.parameter.behaviour_policy.maximum_likelihood
                            if hasattr(
                                config.parameter.behaviour_policy, "maximum_likelihood"
                            )
                            else False
                        ),
                    )
                    behaviour_policy_optimizer.zero_grad()
                    behaviour_policy_loss.backward()
                    if hasattr(config.parameter.behaviour_policy, "grad_norm_clip"):
                        behaviour_model_grad_norms = nn.utils.clip_grad_norm_(
                            self.model["GPPolicy"].base_model.parameters(),
                            max_norm=config.parameter.behaviour_policy.grad_norm_clip,
                            norm_type=2,
                        )
                    behaviour_policy_optimizer.step()

                    counter += 1
                    behaviour_policy_loss_sum += behaviour_policy_loss.item()

                    behaviour_policy_train_iter += 1
                    self.behaviour_policy_train_epoch = epoch

                wandb.log(
                    data=dict(
                        behaviour_policy_train_iter=behaviour_policy_train_iter,
                        behaviour_policy_train_epoch=epoch,
                        behaviour_policy_loss=behaviour_policy_loss_sum / counter,
                        behaviour_model_grad_norms=(
                            behaviour_model_grad_norms.item()
                            if hasattr(
                                config.parameter.behaviour_policy, "grad_norm_clip"
                            )
                            else 0.0
                        ),
                    ),
                    commit=True,
                )

                if (
                    hasattr(config.parameter.behaviour_policy, "lr_decy")
                    and config.parameter.behaviour_policy.lr_decy is True
                ):
                    behaviour_lr_scheduler.step()
                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        iteration=behaviour_policy_train_iter,
                        model_type="base_model",
                    )

            # ---------------------------------------
            # behavior training code ↑
            # ---------------------------------------

            if (
                hasattr(config.parameter, "need_fake_action")
                and config.parameter.need_fake_action is True
            ):
                self.need_fake_action = True
            if config.parameter.algorithm_type in ["GPO_softmax_static"]:
                self.need_fake_action = True
            else:
                self.need_fake_action = False

            # ---------------------------------------
            # make fake action ↓
            # ---------------------------------------

            self.model["GPPolicy"].base_model.eval()
            if self.need_fake_action:

                fake_actions = generate_fake_action(
                    self.model["GPPolicy"],
                    self.dataset.states[:],
                    config.parameter.sample_per_state,
                )
                fake_next_actions = generate_fake_action(
                    self.model["GPPolicy"],
                    self.dataset.next_states[:],
                    config.parameter.sample_per_state,
                )

                log.info(
                    f"Test fake actions: {fake_actions[0].detach().cpu().numpy().mean()}"
                )

                log.info(
                    f"Test fake next actions: {fake_next_actions[0].detach().cpu().numpy().mean()}"
                )

                self.dataset.fake_actions = fake_actions
                self.dataset.fake_next_actions = fake_next_actions

                filename = os.path.join(
                    config.parameter.checkpoint_path,
                    f"dataset_with_fakeaction.npz",
                )
                fake_data = dict(
                    fake_actions=fake_actions.cpu().numpy(),
                    fake_next_actions=fake_next_actions.cpu().numpy(),
                )
                np.savez(filename, **fake_data)

            # ---------------------------------------
            # make fake action ↑
            # ---------------------------------------

            # ---------------------------------------
            # critic training code ↓
            # ---------------------------------------

            if config.parameter.critic.method == "iql":
                v_optimizer = torch.optim.Adam(
                    self.vf.parameters(),
                    lr=config.parameter.critic.learning_rate,
                )

            q_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            if (
                hasattr(config.parameter.critic, "lr_decy")
                and config.parameter.critic.lr_decy is True
            ):
                critic_lr_scheduler = CosineAnnealingLR(
                    q_optimizer,
                    T_max=config.parameter.critic_policy.epochs,
                    eta_min=0.0,
                )

            critic_train_iter = 0
            for epoch in track(
                range(config.parameter.critic.epochs), description="Critic training"
            ):
                if self.critic_train_epoch >= epoch:
                    if (
                        hasattr(config.parameter.critic, "lr_decy")
                        and config.parameter.critic.lr_decy is True
                    ):
                        critic_lr_scheduler.step()
                    continue

                sampler = torch.utils.data.RandomSampler(
                    self.dataset, replacement=False
                )
                data_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=config.parameter.critic.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    pin_memory=False,
                    drop_last=True,
                )

                counter = 1
                if config.parameter.critic.method == "iql":
                    v_loss_sum = 0.0
                    v_sum = 0.0
                q_loss_sum = 0.0
                q_sum = 0.0
                q_target_sum = 0.0
                q_grad_norms_sum = 0.0
                for data in data_loader:

                    if config.parameter.critic.method == "iql":
                        v_loss, next_v = self.model["GPPolicy"].critic.v_loss(
                            self.vf,
                            data,
                            config.parameter.critic.tau,
                        )
                        v_optimizer.zero_grad(set_to_none=True)
                        v_loss.backward()
                        v_optimizer.step()
                        q_loss, q, q_target = self.model["GPPolicy"].critic.iql_q_loss(
                            data,
                            next_v,
                            config.parameter.critic.discount_factor,
                        )
                        q_optimizer.zero_grad(set_to_none=True)
                        q_loss.backward()
                        q_optimizer.step()
                    else:
                        q_loss, q, q_target = self.model["GPPolicy"].q_loss(
                            data["a"],
                            data["s"],
                            data["r"],
                            data["s_"],
                            data["d"],
                            data["fake_a_"],
                            discount_factor=config.parameter.critic.discount_factor,
                        )

                        q_optimizer.zero_grad()
                        q_loss.backward()
                        if hasattr(config.parameter.critic, "grad_norm_clip"):
                            q_grad_norms = nn.utils.clip_grad_norm_(
                                self.model["GPPolicy"].critic.parameters(),
                                max_norm=config.parameter.critic.grad_norm_clip,
                                norm_type=2,
                            )
                        q_optimizer.step()

                    # Update target
                    for param, target_param in zip(
                        self.model["GPPolicy"].critic.parameters(),
                        self.model["GPPolicy"].critic.q_target.parameters(),
                    ):
                        target_param.data.copy_(
                            config.parameter.critic.update_momentum * param.data
                            + (1 - config.parameter.critic.update_momentum)
                            * target_param.data
                        )

                    counter += 1
                    if config.parameter.critic.method == "iql":
                        v_loss_sum += v_loss.item()
                        v_sum += next_v.mean().item()
                    q_loss_sum += q_loss.item()
                    q_sum += q.mean().item()
                    q_target_sum += q_target.mean().item()
                    if hasattr(config.parameter.critic, "grad_norm_clip"):
                        q_grad_norms_sum += q_grad_norms.item()

                    critic_train_iter += 1
                    self.critic_train_epoch = epoch

                if config.parameter.critic.method == "iql":
                    wandb.log(
                        data=dict(v_loss=v_loss_sum / counter, v=v_sum / counter),
                        commit=False,
                    )

                wandb.log(
                    data=dict(
                        critic_train_iter=critic_train_iter,
                        critic_train_epoch=epoch,
                        q_loss=q_loss_sum / counter,
                        q=q_sum / counter,
                        q_target=q_target_sum / counter,
                        q_grad_norms=(
                            q_grad_norms_sum / counter
                            if hasattr(config.parameter.critic, "grad_norm_clip")
                            else 0.0
                        ),
                    ),
                    commit=True,
                )

                if (
                    hasattr(config.parameter.critic, "lr_decy")
                    and config.parameter.critic.lr_decy is True
                ):
                    critic_lr_scheduler.step()

                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    if config.parameter.critic.method == "iql":
                        save_checkpoint(
                            self.model,
                            iteration=critic_train_iter,
                            model_type="critic_model",
                        )
            # ---------------------------------------
            # critic training code ↑
            # ---------------------------------------

            # ---------------------------------------
            # guided policy training code ↓
            # ---------------------------------------

            if not self.guided_policy_train_epoch > 0:
                if (
                    hasattr(config.parameter.guided_policy, "copy_from_basemodel")
                    and config.parameter.guided_policy.copy_from_basemodel
                ):
                    self.model["GPPolicy"].guided_model.model.load_state_dict(
                        self.model["GPPolicy"].base_model.model.state_dict()
                    )

                    for param, target_param in zip(
                        self.model["GPPolicy"].guided_model.model.parameters(),
                        self.model["GPPolicy"].base_model.model.parameters(),
                    ):
                        assert torch.equal(
                            param, target_param
                        ), f"The model is not copied correctly."

            guided_policy_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].guided_model.parameters(),
                lr=config.parameter.guided_policy.learning_rate,
            )

            if (
                hasattr(config.parameter.guided_policy, "lr_decy")
                and config.parameter.guided_policy.lr_decy is True
            ):

                guided_lr_scheduler = CosineAnnealingLR(
                    guided_policy_optimizer,
                    T_max=config.parameter.guided_policy.epochs,
                    eta_min=0.0,
                )

            guided_policy_train_iter = 0

            if hasattr(config.parameter.guided_policy, "eta"):
                eta = config.parameter.guided_policy.eta
            else:
                eta = 1.0

            for epoch in track(
                range(config.parameter.guided_policy.epochs),
                description="Guided policy training",
            ):

                if self.guided_policy_train_epoch >= epoch:
                    if (
                        hasattr(config.parameter.guided_policy, "lr_decy")
                        and config.parameter.guided_policy.lr_decy is True
                    ):
                        guided_lr_scheduler.step()
                    continue

                sampler = torch.utils.data.RandomSampler(
                    self.dataset, replacement=False
                )
                data_loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=config.parameter.guided_policy.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    pin_memory=False,
                    drop_last=True,
                )

                if (
                    hasattr(config.parameter.evaluation, "eval")
                    and config.parameter.evaluation.eval
                ):
                    if (
                        epoch
                        % config.parameter.evaluation.evaluation_guided_policy_interval
                        == 0
                        or (epoch + 1) == config.parameter.guided_policy.epochs
                    ):
                        evaluation_results = evaluate(
                            self.model,
                            train_epoch=epoch,
                            guidance_scales=config.parameter.evaluation.guidance_scale,
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )
                        wandb.log(data=evaluation_results, commit=False)

                counter = 1
                guided_policy_loss_sum = 0.0
                guided_model_grad_norms_sum = 0.0
                if config.parameter.algorithm_type == "GPO":
                    weight_sum = 0.0
                    clamped_weight_sum = 0.0
                    clamped_ratio_sum = 0.0
                elif config.parameter.algorithm_type in [
                    "GPO_softmax_static",
                    "GPO_softmax_sample",
                ]:
                    energy_sum = 0.0
                    relative_energy_sum = 0.0
                    matching_loss_sum = 0.0
                for data in data_loader:
                    if config.parameter.algorithm_type == "GPO":
                        (
                            guided_policy_loss,
                            weight,
                            clamped_weight,
                            clamped_ratio,
                        ) = self.model["GPPolicy"].policy_loss(
                            data["a"],
                            data["s"],
                            data["fake_a"],
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            eta=eta,
                            regularize_method=(
                                config.parameter.guided_policy.regularize_method
                                if hasattr(
                                    config.parameter.guided_policy, "regularize_method"
                                )
                                else "minus_value"
                            ),
                            value_function=(
                                self.vf
                                if config.parameter.critic.method == "iql"
                                else None
                            ),
                            weight_clamp=(
                                config.parameter.guided_policy.weight_clamp
                                if hasattr(
                                    config.parameter.guided_policy, "weight_clamp"
                                )
                                else 100.0
                            ),
                        )
                        weight_sum += weight
                        clamped_weight_sum += clamped_weight
                        clamped_ratio_sum += clamped_ratio
                    elif config.parameter.algorithm_type == "GPO_softmax_static":
                        (
                            guided_policy_loss,
                            energy,
                            relative_energy,
                            matching_loss,
                        ) = self.model["GPPolicy"].policy_loss_softmax(
                            data["s"],
                            data["fake_a"],
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            eta=eta,
                        )
                        energy_sum += energy
                        relative_energy_sum += relative_energy
                        matching_loss_sum += matching_loss
                    elif config.parameter.algorithm_type == "GPO_softmax_sample":
                        fake_actions_ = self.model["GPPolicy"].behaviour_policy_sample(
                            state=data["s"],
                            t_span=(
                                torch.linspace(0.0, 1.0, config.parameter.t_span).to(
                                    data["s"].device
                                )
                                if hasattr(config.parameter, "t_span")
                                and config.parameter.t_span is not None
                                else None
                            ),
                            batch_size=config.parameter.sample_per_state,
                        )
                        fake_actions_ = torch.einsum("nbd->bnd", fake_actions_)
                        (
                            guided_policy_loss,
                            energy,
                            relative_energy,
                            matching_loss,
                        ) = self.model["GPPolicy"].policy_loss_softmax(
                            data["s"],
                            fake_actions_,
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            eta=eta,
                        )
                        energy_sum += energy
                        relative_energy_sum += relative_energy
                        matching_loss_sum += matching_loss
                    elif config.parameter.algorithm_type == "GPG_Direct":
                        guided_policy_loss = self.model[
                            "GPPolicy"
                        ].policy_loss_pure_grad(
                            data["s"],
                            loss_type=config.parameter.guided_policy.loss_type,
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            eta=eta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                            value_function=(
                                self.vf
                                if config.parameter.critic.method == "iql"
                                else None
                            ),
                        )
                    elif config.parameter.algorithm_type == "GPG_Polish":
                        (
                            guided_policy_loss,
                            eta_q_loss,
                            log_p_loss,
                            log_u_loss,
                        ) = self.model["GPPolicy"].policy_loss_pure_grad_polish(
                            data["s"],
                            loss_type=config.parameter.guided_policy.loss_type,
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            eta=eta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                            value_function=(
                                self.vf
                                if config.parameter.critic.method == "iql"
                                else None
                            ),
                            weight_clamp=(
                                config.parameter.guided_policy.weight_clamp
                                if hasattr(
                                    config.parameter.guided_policy, "weight_clamp"
                                )
                                else 100.0
                            ),
                        )
                    elif config.parameter.algorithm_type == "GPG_Softmax":
                        (
                            guided_policy_loss,
                            eta_q_loss,
                            log_p_loss,
                            log_u_loss,
                        ) = self.model["GPPolicy"].policy_loss_pure_grad_softmax(
                            data["s"],
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            eta=eta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 32
                            ),
                        )
                    elif config.parameter.algorithm_type == "GPG_2":
                        guided_policy_loss = self.model[
                            "GPPolicy"
                        ].policy_loss_withgrade(
                            data["a"],
                            data["s"],
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            loss_type=config.parameter.guided_policy.loss_type,
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            eta=eta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                        )
                    else:
                        raise NotImplementedError
                    guided_policy_optimizer.zero_grad()
                    guided_policy_loss.backward()
                    if hasattr(config.parameter.guided_policy, "grad_norm_clip"):
                        guided_model_grad_norms = nn.utils.clip_grad_norm_(
                            self.model["GPPolicy"].guided_model.parameters(),
                            max_norm=config.parameter.guided_policy.grad_norm_clip,
                            norm_type=2,
                        )
                    else:
                        guided_model_grad_norms = nn.utils.clip_grad_norm_(
                            self.model["GPPolicy"].guided_model.parameters(),
                            max_norm=100000,
                            norm_type=2,
                        )
                    guided_policy_optimizer.step()
                    if config.parameter.algorithm_type == "GPG_Direct":
                        wandb.log(
                            data=dict(
                                guided_policy_train_iter=guided_policy_train_iter,
                                guided_policy_train_epoch=epoch,
                                guided_policy_loss=guided_policy_loss.item(),
                                guided_model_grad_norms=guided_model_grad_norms,
                            ),
                            commit=True,
                        )
                        save_checkpoint(
                            self.model,
                            iteration=guided_policy_train_iter,
                            model_type="guided_model",
                        )
                    elif config.parameter.algorithm_type in [
                        "GPG_Polish",
                        "GPG_Softmax",
                    ]:
                        wandb.log(
                            data=dict(
                                guided_policy_train_iter=guided_policy_train_iter,
                                guided_policy_train_epoch=epoch,
                                guided_policy_loss=guided_policy_loss.item(),
                                guided_model_grad_norms=guided_model_grad_norms,
                                eta_q_loss=eta_q_loss.item(),
                                log_p_loss=log_p_loss.item(),
                                log_u_loss=log_u_loss.item(),
                            ),
                            commit=True,
                        )
                        save_checkpoint(
                            self.model,
                            iteration=guided_policy_train_iter,
                            model_type="guided_model",
                        )
                    counter += 1
                    guided_policy_loss_sum += guided_policy_loss.item()
                    if hasattr(config.parameter.guided_policy, "grad_norm_clip"):
                        guided_model_grad_norms_sum += guided_model_grad_norms.item()

                    guided_policy_train_iter += 1
                    self.guided_policy_train_epoch = epoch
                    if (
                        config.parameter.evaluation.eval
                        and hasattr(
                            config.parameter.evaluation, "evaluation_iteration_interval"
                        )
                        and (guided_policy_train_iter + 1)
                        % config.parameter.evaluation.evaluation_iteration_interval
                        == 0
                    ):
                        evaluation_results = evaluate(
                            self.model,
                            train_epoch=epoch,
                            guidance_scales=config.parameter.evaluation.guidance_scale,
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )
                        wandb.log(data=evaluation_results, commit=False)
                        wandb.log(
                            data=dict(
                                guided_policy_train_iter=guided_policy_train_iter,
                                guided_policy_train_epoch=epoch,
                            ),
                            commit=True,
                        )
                if config.parameter.algorithm_type in [
                    "GPO",
                    "GPO_softmax_static",
                    "GPO_softmax_sample",
                ]:
                    if config.parameter.algorithm_type == "GPO":
                        wandb.log(
                            data=dict(
                                weight=weight_sum / counter,
                                clamped_weight=clamped_weight_sum / counter,
                                clamped_ratio=clamped_ratio_sum / counter,
                            ),
                            commit=False,
                        )
                    elif config.parameter.algorithm_type in [
                        "GPO_softmax_static",
                        "GPO_softmax_sample",
                    ]:
                        wandb.log(
                            data=dict(
                                energy=energy_sum / counter,
                                relative_energy=relative_energy_sum / counter,
                                matching_loss=matching_loss_sum / counter,
                            ),
                            commit=False,
                        )

                    wandb.log(
                        data=dict(
                            guided_policy_train_iter=guided_policy_train_iter,
                            guided_policy_train_epoch=epoch,
                            guided_policy_loss=guided_policy_loss_sum / counter,
                            guided_model_grad_norms=(
                                guided_model_grad_norms_sum / counter
                                if hasattr(
                                    config.parameter.guided_policy, "grad_norm_clip"
                                )
                                else 0.0
                            ),
                        ),
                        commit=True,
                    )

                if (
                    hasattr(config.parameter.guided_policy, "lr_decy")
                    and config.parameter.guided_policy.lr_decy is True
                ):
                    guided_lr_scheduler.step()

                if (
                    hasattr(config.parameter, "checkpoint_guided_freq")
                    and (epoch + 1) % config.parameter.checkpoint_guided_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        iteration=guided_policy_train_iter,
                        model_type="guided_model",
                    )
                elif (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
                    save_checkpoint(
                        self.model,
                        iteration=guided_policy_train_iter,
                        model_type="guided_model",
                    )

            # ---------------------------------------
            # guided policy training code ↑
            # ---------------------------------------

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

        wandb.finish()

    def deploy(self, config: EasyDict = None) -> GPAgent:

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "GPPolicy" in self.model, "The model must be trained first."
        assert "GuidedPolicy" in self.model, "The model must be trained first."
        return GPAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
