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
from grl.datasets.gp import GPDataset, GPD4RLDataset, GPD4RLTensorDictDataset
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
from grl.utils.plot import plot_distribution, plot_histogram2d_x_y
from grl.utils.statistics import sort_files_by_criteria
from grl.generative_models.metric import compute_likelihood


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class GMPOCritic(nn.Module):
    """
    Overview:
        Critic network for GMPO algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of GMPO critic network.
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
            Return the output of GMPO critic.
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


class GMPOPolicy(nn.Module):
    """
    Overview:
        GMPO policy network for GMPO algorithm, which includes the base model (optinal), the guided model and the critic.
    Interfaces:
        ``__init__``, ``forward``, ``sample``, ``compute_q``, ``behaviour_policy_loss``, ``policy_optimization_loss_by_advantage_weighted_regression``, ``policy_optimization_loss_by_advantage_weighted_regression_softmax``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the GMPO policy network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = GMPOCritic(config.critic)
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
            Return the output of GMPO policy, which is the action conditioned on the state.
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
            Return the output of GMPO policy, which is the action conditioned on the state.
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

    def policy_optimization_loss_by_advantage_weighted_regression(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
        beta: float = 1.0,
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

        with torch.no_grad():
            q_value = self.critic(action, state).squeeze(dim=-1)
            v_value = self.critic.v(state).squeeze(dim=-1)
            weight = torch.exp(beta * (q_value - v_value))

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

    def policy_optimization_loss_by_advantage_weighted_regression_softmax(
        self,
        state: Union[torch.Tensor, TensorDict],
        fake_action: Union[torch.Tensor, TensorDict],
        maximum_likelihood: bool = False,
        beta: float = 1.0,
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

        relative_energy = nn.Softmax(dim=1)(energy * beta)

        loss = torch.mean(torch.sum(relative_energy * model_loss, axis=-1), dim=1)

        return (
            loss,
            torch.mean(energy),
            torch.mean(relative_energy),
            torch.mean(model_loss),
        )


class GMPOAlgorithm:
    """
    Overview:
        The Generative Model Policy Optimization(GMPO) algorithm.
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
            Initialize the GMPO && GPG algorithm.
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
                    GMPOPolicy(config.model.GPPolicy).to(config.model.GPPolicy.device)
                )
            else:
                self.model["GPPolicy"] = GMPOPolicy(config.model.GPPolicy).to(
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
                "GMPO",
                "GMPO_softmax_static",
                "GMPO_softmax_sample",
            ]
            run_name = f"{config.parameter.critic.method}-tau-{config.parameter.critic.tau}-beta-{config.parameter.guided_policy.beta}-batch-{config.parameter.guided_policy.batch_size}-lr-{config.parameter.guided_policy.learning_rate}-{config.model.GPPolicy.model.model.type}-{self.seed_value}"
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
                    if isinstance(obs, np.ndarray):
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
                                device=config.model.GPPolicy.device,
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

                if isinstance(self.dataset, GPD4RLDataset) or isinstance(
                    self.dataset, GPD4RLTensorDictDataset
                ):
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

            replay_buffer = TensorDictReplayBuffer(
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

                if (
                    hasattr(config.parameter.evaluation, "analysis_interval")
                    and epoch % config.parameter.evaluation.analysis_interval == 0
                ):
                    for index, data in enumerate(replay_buffer):

                        evaluation_results = evaluate(
                            self.model["GPPolicy"].base_model,
                            train_epoch=epoch,
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )
                        if not os.path.exists(config.parameter.checkpoint_path):
                            os.makedirs(config.parameter.checkpoint_path)
                        plot_distribution(
                            data["a"],
                            os.path.join(
                                config.parameter.checkpoint_path,
                                f"action_base_{epoch}.png",
                            ),
                        )

                        action = self.model["GPPolicy"].sample(
                            state=data["s"].to(config.model.GPPolicy.device),
                            t_span=(
                                torch.linspace(0.0, 1.0, config.parameter.t_span).to(
                                    config.model.GPPolicy.device
                                )
                                if hasattr(config.parameter, "t_span")
                                and config.parameter.t_span is not None
                                else None
                            ),
                        )
                        plot_distribution(
                            action,
                            os.path.join(
                                config.parameter.checkpoint_path,
                                f"action_base_model_{epoch}_{evaluation_results['evaluation/return_mean']}.png",
                            ),
                        )

                        wandb.log(data=evaluation_results, commit=False)
                        break

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
            # make fake action ↓
            # ---------------------------------------

            if config.parameter.algorithm_type in ["GMPO_softmax_static"]:
                data_augmentation = True
            else:
                data_augmentation = False

            self.model["GPPolicy"].base_model.eval()
            if data_augmentation:

                fake_actions = generate_fake_action(
                    self.model["GPPolicy"],
                    self.dataset.states[:].to(config.model.GPPolicy.device),
                    config.parameter.action_augment_num,
                )
                fake_next_actions = generate_fake_action(
                    self.model["GPPolicy"],
                    self.dataset.next_states[:].to(config.model.GPPolicy.device),
                    config.parameter.action_augment_num,
                )

                self.dataset.load_fake_actions(
                    fake_actions=fake_actions.to("cpu"),
                    fake_next_actions=fake_next_actions.to("cpu"),
                )

            # ---------------------------------------
            # make fake action ↑
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

            replay_buffer = TensorDictReplayBuffer(
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
                        self.model["GPPolicy"].critic.q.parameters(),
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
                if (
                    hasattr(config.parameter.guided_policy, "copy_from_basemodel")
                    and config.parameter.guided_policy.copy_from_basemodel
                ):
                    self.model["GPPolicy"].guided_model.model.load_state_dict(
                        self.model["GPPolicy"].base_model.model.state_dict()
                    )

            guided_policy_optimizer = torch.optim.Adam(
                self.model["GPPolicy"].guided_model.parameters(),
                lr=config.parameter.guided_policy.learning_rate,
            )
            guided_policy_train_iter = 0
            logp_mean = []
            end_return = []
            beta = config.parameter.guided_policy.beta

            replay_buffer = TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.guided_policy.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            for epoch in track(
                range(config.parameter.guided_policy.epochs),
                description="Guided policy training",
            ):

                if self.guided_policy_train_epoch >= epoch:
                    continue

                if (
                    hasattr(config.parameter.evaluation, "analysis_interval")
                    and epoch % config.parameter.evaluation.analysis_interval == 0
                ):
                    timlimited = 0
                    for index, data in enumerate(replay_buffer):
                        if timlimited == 0:
                            if not os.path.exists(config.parameter.checkpoint_path):
                                os.makedirs(config.parameter.checkpoint_path)
                            plot_distribution(
                                data["a"].detach().cpu().numpy(),
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"action_base_{epoch}.png",
                                ),
                            )

                            action = self.model["GPPolicy"].sample(
                                state=data["s"].to(config.model.GPPolicy.device),
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.t_span
                                    ).to(config.model.GPPolicy.device)
                                    if hasattr(config.parameter, "t_span")
                                    and config.parameter.t_span is not None
                                    else None
                                ),
                            )

                        evaluation_results = evaluate(
                            self.model["GPPolicy"].guided_model,
                            train_epoch=epoch,
                            repeat=(
                                1
                                if not hasattr(config.parameter.evaluation, "repeat")
                                else config.parameter.evaluation.repeat
                            ),
                        )

                        log_p = compute_likelihood(
                            model=self.model["GPPolicy"].guided_model,
                            x=data["a"].to(config.model.GPPolicy.device),
                            condition=data["s"].to(config.model.GPPolicy.device),
                            t=torch.linspace(0.0, 1.0, 100).to(
                                config.model.GPPolicy.device
                            ),
                            using_Hutchinson_trace_estimator=True,
                        )
                        logp_mean.append(log_p.mean().detach().cpu().numpy())
                        end_return.append(evaluation_results["evaluation/return_mean"])

                        if timlimited == 0:
                            plot_distribution(
                                action.detach().cpu().numpy(),
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"action_guided_model_{epoch}_{evaluation_results['evaluation/return_mean']}.png",
                                ),
                            )
                        timlimited += 1
                        wandb.log(data=evaluation_results, commit=False)

                        if timlimited > 10:
                            logp_dict = {
                                "logp_mean": logp_mean,
                                "end_return": end_return,
                            }
                            np.savez(
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"logp_data_guided_{epoch}.npz",
                                ),
                                **logp_dict,
                            )
                            plot_histogram2d_x_y(
                                end_return,
                                logp_mean,
                                os.path.join(
                                    config.parameter.checkpoint_path,
                                    f"return_logp_guided_{epoch}.png",
                                ),
                            )
                            break

                counter = 1
                guided_policy_loss_sum = 0.0
                if config.parameter.algorithm_type == "GMPO":
                    weight_sum = 0.0
                    clamped_weight_sum = 0.0
                    clamped_ratio_sum = 0.0
                elif config.parameter.algorithm_type in [
                    "GMPO_softmax_static",
                    "GMPO_softmax_sample",
                ]:
                    energy_sum = 0.0
                    relative_energy_sum = 0.0
                    matching_loss_sum = 0.0

                for index, data in enumerate(replay_buffer):
                    if config.parameter.algorithm_type == "GMPO":
                        (
                            guided_policy_loss,
                            weight,
                            clamped_weight,
                            clamped_ratio,
                        ) = self.model[
                            "GPPolicy"
                        ].policy_optimization_loss_by_advantage_weighted_regression(
                            data["a"].to(config.model.GPPolicy.device),
                            data["s"].to(config.model.GPPolicy.device),
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            beta=beta,
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
                    elif config.parameter.algorithm_type == "GMPO_softmax_static":
                        (
                            guided_policy_loss,
                            energy,
                            relative_energy,
                            matching_loss,
                        ) = self.model[
                            "GPPolicy"
                        ].policy_optimization_loss_by_advantage_weighted_regression_softmax(
                            data["s"].to(config.model.GPPolicy.device),
                            data["fake_a"].to(config.model.GPPolicy.device),
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            beta=beta,
                        )
                        energy_sum += energy
                        relative_energy_sum += relative_energy
                        matching_loss_sum += matching_loss
                    elif config.parameter.algorithm_type == "GMPO_softmax_sample":
                        fake_actions_ = self.model["GPPolicy"].behaviour_policy_sample(
                            state=data["s"].to(config.model.GPPolicy.device),
                            t_span=(
                                torch.linspace(0.0, 1.0, config.parameter.t_span).to(
                                    config.model.GPPolicy.device
                                )
                                if hasattr(config.parameter, "t_span")
                                and config.parameter.t_span is not None
                                else None
                            ),
                            batch_size=config.parameter.action_augment_num,
                        )
                        fake_actions_ = torch.einsum("nbd->bnd", fake_actions_)
                        (
                            guided_policy_loss,
                            energy,
                            relative_energy,
                            matching_loss,
                        ) = self.model[
                            "GPPolicy"
                        ].policy_optimization_loss_by_advantage_weighted_regression_softmax(
                            data["s"].to(config.model.GPPolicy.device),
                            fake_actions_,
                            maximum_likelihood=(
                                config.parameter.guided_policy.maximum_likelihood
                                if hasattr(
                                    config.parameter.guided_policy, "maximum_likelihood"
                                )
                                else False
                            ),
                            beta=beta,
                        )
                        energy_sum += energy
                        relative_energy_sum += relative_energy
                        matching_loss_sum += matching_loss
                    else:
                        raise NotImplementedError
                    guided_policy_optimizer.zero_grad()
                    guided_policy_loss.backward()
                    guided_policy_optimizer.step()
                    counter += 1

                    guided_policy_loss_sum += guided_policy_loss.item()

                    guided_policy_train_iter += 1
                    self.guided_policy_train_epoch = epoch

                if (
                    config.parameter.evaluation.eval
                    and hasattr(config.parameter.evaluation, "epoch_interval")
                    and (self.guided_policy_train_epoch + 1)
                    % config.parameter.evaluation.epoch_interval
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

                if config.parameter.algorithm_type == "GMPO":
                    wandb.log(
                        data=dict(
                            weight=weight_sum / counter,
                            clamped_weight=clamped_weight_sum / counter,
                            clamped_ratio=clamped_ratio_sum / counter,
                        ),
                        commit=False,
                    )
                elif config.parameter.algorithm_type in [
                    "GMPO_softmax_static",
                    "GMPO_softmax_sample",
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
                    ),
                    commit=True,
                )

                if (
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
        return GPAgent(
            config=config,
            model=copy.deepcopy(self.model["GPPolicy"].guided_model),
        )
