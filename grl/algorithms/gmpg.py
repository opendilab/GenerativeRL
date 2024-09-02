import os
import copy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import track
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

import wandb
from grl.agents.gm import GPAgent

from grl.datasets import create_dataset
from grl.datasets.gp import GPDataset, GPD4RLDataset
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

from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.rl_modules.value_network.value_network import VNetwork, DoubleVNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.utils import set_seed
from grl.utils.statistics import sort_files_by_criteria
from grl.generative_models.metric import compute_likelihood


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class GMPGCritic(nn.Module):
    """
    Overview:
        Critic network.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of GPO critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.q_alpha = config.q_alpha
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.v = VNetwork(config.VNetwork)

    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of GPO critic.
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

    def in_support_ql_loss(
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

    def v_loss(self, state, action, next_state, tau):
        with torch.no_grad():
            target_q = self.q_target(action, state).detach()
            next_v = self.v(next_state).detach()
        # Update value function
        v = self.v(state)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, tau)
        return v_loss, next_v

    def iql_q_loss(self, state, action, reward, done, next_v, discount):
        q_target = reward + (1.0 - done.float()) * discount * next_v.detach()
        qs = self.q.compute_double_q(action, state)
        q_loss = sum(torch.nn.functional.mse_loss(q, q_target) for q in qs) / len(qs)
        return q_loss, torch.mean(qs[0]), torch.mean(q_target)


class GMPGPolicy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = GMPGCritic(config.critic)
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
            Return the output of GPO policy, which is the action conditioned on the state.
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
            Return the output of GPO policy, which is the action conditioned on the state.
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

    def policy_gradient_loss(
        self,
        state: Union[torch.Tensor, TensorDict],
        gradtime_step: int = 1000,
        beta: float = 1.0,
        repeats: int = 1,
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

        q_value_repeated = self.critic(action_repeated, state_repeated).squeeze(dim=-1)
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
        log_p_per_dim = log_p / bits_ratio
        log_mu = compute_likelihood(
            model=self.base_model,
            x=action_repeated,
            condition=state_repeated,
            t=t_span,
            using_Hutchinson_trace_estimator=True,
        )
        log_mu.register_hook(lambda grad: log_grad("log_mu", grad))
        log_mu_per_dim = log_mu / bits_ratio

        if repeats > 1:
            q_value_repeated = q_value_repeated.reshape(-1, repeats)
            log_p_per_dim = log_p_per_dim.reshape(-1, repeats)
            log_mu_per_dim = log_mu_per_dim.reshape(-1, repeats)

            return (
                (
                    -beta * q_value_repeated.mean(dim=1)
                    + log_p_per_dim(dim=1)
                    - log_mu_per_dim(dim=1)
                ),
                -beta * q_value_repeated.detach().mean(),
                log_p_per_dim.detach().mean(),
                -log_mu_per_dim.detach().mean(),
            )
        else:
            return (
                (-beta * q_value_repeated + log_p_per_dim - log_mu_per_dim).mean(),
                -beta * q_value_repeated.detach().mean(),
                log_p_per_dim.detach().mean(),
                -log_mu_per_dim.detach().mean(),
            )

    def policy_gradient_loss_by_REINFORCE(
        self,
        state: Union[torch.Tensor, TensorDict],
        gradtime_step: int = 1000,
        beta: float = 1.0,
        repeats: int = 1,
        weight_clamp: float = 100.0,
    ):
        t_span = torch.linspace(0.0, 1.0, gradtime_step).to(state.device)

        state_repeated = torch.repeat_interleave(state, repeats=repeats, dim=0)
        action_repeated = self.base_model.sample(
            t_span=t_span, condition=state_repeated, with_grad=False
        )
        q_value_repeated = self.critic(action_repeated, state_repeated).squeeze(dim=-1)
        v_value_repeated = self.critic.v(state_repeated).squeeze(dim=-1)

        weight = (
            torch.exp(beta * (q_value_repeated - v_value_repeated)).clamp(
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
                -beta * q_value_repeated.detach()
                + log_p_per_dim.detach()
                - log_mu_per_dim.detach()
            )
            * log_p_per_dim
            * weight
        )
        with torch.no_grad():
            loss_q = -beta * q_value_repeated.detach().mean()
            loss_p = log_p_per_dim.detach().mean()
            loss_u = -log_mu_per_dim.detach().mean()
        return loss, loss_q, loss_p, loss_u

    def policy_gradient_loss_by_REINFORCE_softmax(
        self,
        state: Union[torch.Tensor, TensorDict],
        gradtime_step: int = 1000,
        beta: float = 1.0,
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

        weight = nn.Softmax(dim=1)(q_value_reshaped * beta)
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
                -beta * q_value_repeated.detach()
                + log_p_per_dim.detach()
                - log_mu_per_dim.detach()
            )
            * log_p_per_dim
            * weight
        )
        loss_q = -beta * q_value_repeated.detach().mean()
        loss_p = log_p_per_dim.detach().mean()
        loss_u = -log_mu_per_dim.detach().mean()
        return loss, loss_q, loss_p, loss_u


class GMPGAlgorithm:
    """
    Overview:
        The Generative Model Policy Gradient(GMPG) algorithm.
    Interfaces:
        ``__init__``, ``train``, ``deploy``
    """

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
            Initialize algorithm.
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
        self.seed_value = set_seed(seed)

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
                    GMPGPolicy(config.model.GPPolicy).to(config.model.GPPolicy.device)
                )
            else:
                self.model["GPPolicy"] = GMPGPolicy(config.model.GPPolicy).to(
                    config.model.GPPolicy.device
                )

            if (
                hasattr(config.parameter, "checkpoint_path")
                and config.parameter.checkpoint_path is not None
            ):
                if not os.path.exists(config.parameter.checkpoint_path):
                    log.warning(
                        f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
                    )
                    self.behaviour_policy_train_epoch = -1
                    self.critic_train_epoch = -1
                    self.guided_policy_train_epoch = -1
                else:
                    base_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="basemodel_",
                        end_string=".pt",
                    )
                    if len(base_model_files) == 0:
                        self.behaviour_policy_train_epoch = -1
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
                            "behaviour_policy_train_epoch", -1
                        )

                    guided_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="guidedmodel_",
                        end_string=".pt",
                    )
                    if len(guided_model_files) == 0:
                        self.guided_policy_train_epoch = -1
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
                            "guided_policy_train_epoch", -1
                        )

                    critic_model_files = sort_files_by_criteria(
                        folder_path=config.parameter.checkpoint_path,
                        start_string="critic_",
                        end_string=".pt",
                    )
                    if len(critic_model_files) == 0:
                        self.critic_train_epoch = -1
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
                            "critic_train_epoch", -1
                        )

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

        config["seed"] = self.seed_value if seed is None else seed

        if not hasattr(config, "wandb"):
            config["wandb"] = dict(project=config.project)
        elif not hasattr(config.wandb, "project"):
            config.wandb["project"] = config.project

        with wandb.init(**config.wandb) as wandb_run:
            if not hasattr(config.parameter.guided_policy, "beta"):
                config.parameter.guided_policy.beta = 1.0

            assert config.parameter.algorithm_type in [
                "GMPG",
                "GMPG_REINFORCE",
                "GMPG_REINFORCE_softmax",
                "GMPG_add_matching",
            ]
            run_name = f"{config.parameter.critic.method}-beta-{config.parameter.guided_policy.beta}-T-{config.parameter.guided_policy.gradtime_step}-batch-{config.parameter.guided_policy.batch_size}-lr-{config.parameter.guided_policy.learning_rate}-seed-{self.seed_value}"
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
                    else:
                        raise NotImplementedError

            def generate_fake_action(model, states, action_augment_num):

                fake_actions_sampled = []
                for states in track(
                    np.array_split(states, states.shape[0] // 4096 + 1),
                    description="Generate fake actions",
                ):

                    fake_actions_ = model.behaviour_policy_sample(
                        state=states,
                        batch_size=action_augment_num,
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

            def evaluate(model, train_epoch, repeat=1):
                evaluation_results = dict()

                def policy(obs: np.ndarray) -> np.ndarray:
                    if isinstance(obs, torch.Tensor):
                        obs = torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=config.model.GPPolicy.device,
                        ).unsqueeze(0)
                    elif isinstance(obs, dict):
                        for key in obs:
                            obs[key] = torch.tensor(
                                obs[key],
                                dtype=torch.float32,
                                device=config.model.GPPolicy.device
                            ).unsqueeze(0)
                            if obs[key].dim() == 1 and obs[key].shape[0] == 1:
                                obs[key] = obs[key].unsqueeze(1)
                        obs = TensorDict(obs, batch_size=[1])
                    action = (
                        model.sample(
                            condition=obs,
                            t_span=(
                                torch.linspace(0.0, 1.0, config.parameter.t_span).to(
                                    config.model.GPPolicy.device
                                )
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
                evaluation_results[f"evaluation/return_mean"] = return_mean
                evaluation_results[f"evaluation/return_std"] = return_std
                evaluation_results[f"evaluation/return_max"] = return_max
                evaluation_results[f"evaluation/return_min"] = return_min

                if isinstance(self.dataset, GPD4RLDataset):
                    import d4rl
                    env_id = config.dataset.args.env_id
                    evaluation_results[f"evaluation/return_mean_normalized"] = (
                        d4rl.get_normalized_score(env_id, return_mean)
                    )
                    evaluation_results[f"evaluation/return_std_normalized"] = (
                        d4rl.get_normalized_score(env_id, return_std)
                    )
                    evaluation_results[f"evaluation/return_max_normalized"] = (
                        d4rl.get_normalized_score(env_id, return_max)
                    )
                    evaluation_results[f"evaluation/return_min_normalized"] = (
                        d4rl.get_normalized_score(env_id, return_min)
                    )

                if repeat > 1:
                    log.info(
                        f"Train epoch: {train_epoch}, return_mean: {return_mean}, return_std: {return_std}, return_max: {return_max}, return_min: {return_min}"
                    )
                else:
                    log.info(f"Train epoch: {train_epoch}, return: {return_mean}")

                return evaluation_results

            # ---------------------------------------
            # behavior training code ↓
            # ---------------------------------------          
            behaviour_policy_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].base_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            replay_buffer=TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.behaviour_policy.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            behaviour_policy_train_iter = 0
            for epoch in track(
                range(config.parameter.behaviour_policy.epochs),
                description="Behaviour policy training",
            ):
                if self.behaviour_policy_train_epoch >= epoch:
                    continue

                counter = 1
                behaviour_policy_loss_sum = 0
                for index, data in enumerate(replay_buffer):

                    behaviour_policy_loss = self.model[
                        "GPPolicy"
                    ].behaviour_policy_loss(
                        action=data["a"].to(config.model.GPPolicy.device),
                        state=data["s"].to(config.model.GPPolicy.device),
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
                    ),
                    commit=True,
                )

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

            # ---------------------------------------
            # critic training code ↓
            # ---------------------------------------

            q_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )
            v_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].critic.v.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            replay_buffer=TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.critic.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            critic_train_iter = 0
            for epoch in track(
                range(config.parameter.critic.epochs), description="Critic training"
            ):
                if self.critic_train_epoch >= epoch:
                    continue

                counter = 1

                v_loss_sum = 0.0
                v_sum = 0.0
                q_loss_sum = 0.0
                q_sum = 0.0
                q_target_sum = 0.0
                for index, data in enumerate(replay_buffer):

                    v_loss, next_v = self.model["GPPolicy"].critic.v_loss(
                        state=data["s"].to(config.model.GPPolicy.device),
                        action=data["a"].to(config.model.GPPolicy.device),
                        next_state=data["s_"].to(config.model.GPPolicy.device),
                        tau=config.parameter.critic.tau,
                    )
                    v_optimizer.zero_grad(set_to_none=True)
                    v_loss.backward()
                    v_optimizer.step()
                    q_loss, q, q_target = self.model["GPPolicy"].critic.iql_q_loss(
                        state=data["s"].to(config.model.GPPolicy.device),
                        action=data["a"].to(config.model.GPPolicy.device),
                        reward=data["r"].to(config.model.GPPolicy.device),
                        done=data["d"].to(config.model.GPPolicy.device),
                        next_v=next_v,
                        discount=config.parameter.critic.discount_factor,
                    )
                    q_optimizer.zero_grad(set_to_none=True)
                    q_loss.backward()
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

                    q_loss_sum += q_loss.item()
                    q_sum += q.mean().item()
                    q_target_sum += q_target.mean().item()

                    v_loss_sum += v_loss.item()
                    v_sum += next_v.mean().item()

                    critic_train_iter += 1
                    self.critic_train_epoch = epoch

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
                    ),
                    commit=True,
                )

                if (
                    hasattr(config.parameter, "checkpoint_freq")
                    and (epoch + 1) % config.parameter.checkpoint_freq == 0
                ):
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
                self.model["GPPolicy"].guided_model.model.load_state_dict(
                    self.model["GPPolicy"].base_model.model.state_dict()
                )

            guided_policy_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].guided_model.parameters(),
                lr=config.parameter.guided_policy.learning_rate,
            )

            replay_buffer=TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.guided_policy.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            guided_policy_train_iter = 0
            beta = config.parameter.guided_policy.beta
            for epoch in track(
                range(config.parameter.guided_policy.epochs),
                description="Guided policy training",
            ):

                if self.guided_policy_train_epoch >= epoch:
                    continue

                counter = 1
                guided_policy_loss_sum = 0.0
                for index, data in enumerate(replay_buffer):
                    if config.parameter.algorithm_type == "GMPG":
                        (
                            guided_policy_loss,
                            q_loss,
                            log_p_loss,
                            log_u_loss,
                        ) = self.model["GPPolicy"].policy_gradient_loss(
                            data["s"].to(config.model.GPPolicy.device),
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            beta=beta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                        )
                    elif config.parameter.algorithm_type == "GMPG_REINFORCE":
                        (
                            guided_policy_loss,
                            q_loss,
                            log_p_loss,
                            log_u_loss,
                        ) = self.model["GPPolicy"].policy_gradient_loss_by_REINFORCE(
                            data["s"].to(config.model.GPPolicy.device),
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            beta=beta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                            weight_clamp=(
                                config.parameter.guided_policy.weight_clamp
                                if hasattr(
                                    config.parameter.guided_policy, "weight_clamp"
                                )
                                else 100.0
                            ),
                        )
                    elif config.parameter.algorithm_type == "GMPG_REINFORCE_softmax":
                        (
                            guided_policy_loss,
                            q_loss,
                            log_p_loss,
                            log_u_loss,
                        ) = self.model[
                            "GPPolicy"
                        ].policy_gradient_loss_by_REINFORCE_softmax(
                            data["s"].to(config.model.GPPolicy.device),
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            beta=beta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 32
                            ),
                        )
                    elif config.parameter.algorithm_type == "GMPG_add_matching":
                        guided_policy_loss = self.model[
                            "GPPolicy"
                        ].policy_gradient_loss_add_matching_loss(
                            data["a"].to(config.model.GPPolicy.device),
                            data["s"].to(config.model.GPPolicy.device),
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            gradtime_step=config.parameter.guided_policy.gradtime_step,
                            beta=beta,
                            repeats=(
                                config.parameter.guided_policy.repeats
                                if hasattr(config.parameter.guided_policy, "repeats")
                                else 1
                            ),
                        )
                    else:
                        raise NotImplementedError
                    guided_policy_optimizer.zero_grad()
                    guided_policy_loss = guided_policy_loss * (
                        data["s"].shape[0] / config.parameter.guided_policy.batch_size
                    )
                    guided_policy_loss.backward()
                    guided_policy_optimizer.step()
                    counter += 1

                    if config.parameter.algorithm_type == "GMPG_add_matching":
                        wandb.log(
                            data=dict(
                                guided_policy_train_iter=guided_policy_train_iter,
                                guided_policy_train_epoch=epoch,
                                guided_policy_loss=guided_policy_loss.item(),
                            ),
                            commit=False,
                        )
                        if (
                            hasattr(config.parameter, "checkpoint_freq")
                            and (guided_policy_train_iter + 1)
                            % config.parameter.checkpoint_freq
                            == 0
                        ):
                            save_checkpoint(
                                self.model,
                                iteration=guided_policy_train_iter,
                                model_type="guided_model",
                            )

                    elif config.parameter.algorithm_type in [
                        "GMPG",
                        "GMPG_REINFORCE",
                        "GMPG_REINFORCE_softmax",
                    ]:
                        wandb.log(
                            data=dict(
                                guided_policy_train_iter=guided_policy_train_iter,
                                guided_policy_train_epoch=epoch,
                                guided_policy_loss=guided_policy_loss.item(),
                                q_loss=q_loss.item(),
                                log_p_loss=log_p_loss.item(),
                                log_u_loss=log_u_loss.item(),
                            ),
                            commit=False,
                        )
                        if (
                            hasattr(config.parameter, "checkpoint_freq")
                            and (guided_policy_train_iter + 1)
                            % config.parameter.checkpoint_freq
                            == 0
                        ):
                            save_checkpoint(
                                self.model,
                                iteration=guided_policy_train_iter,
                                model_type="guided_model",
                            )

                    guided_policy_loss_sum += guided_policy_loss.item()

                    guided_policy_train_iter += 1
                    self.guided_policy_train_epoch = epoch
                    if (
                        config.parameter.evaluation.eval
                        and hasattr(config.parameter.evaluation, "interval")
                        and guided_policy_train_iter
                        % config.parameter.evaluation.interval
                        == 0
                    ):
                        evaluation_results = evaluate(
                            self.model["GPPolicy"].guided_model,
                            train_epoch=epoch,
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
        return GPAgent(
            config=config,
            model=copy.deepcopy(self.model["GPPolicy"].guided_model),
        )
