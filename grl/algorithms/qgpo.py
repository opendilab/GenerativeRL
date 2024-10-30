#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

import copy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

import wandb
from grl.agents.qgpo import QGPOAgent
from grl.datasets import create_dataset
from grl.datasets.qgpo import QGPODataset
from grl.generative_models.diffusion_model.energy_conditional_diffusion_model import (
    EnergyConditionalDiffusionModel,
)
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log
from grl.utils.model_utils import save_model, load_model


class QGPOCritic(nn.Module):
    """
    Overview:
        Critic network for QGPO algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of QGPO critic network.
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
            Return the output of QGPO critic.
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
            if isinstance(next_state, TensorDict):
                new_next_state = next_state.clone(False)
                for key, value in next_state.items():
                    if isinstance(value, torch.Tensor):
                        stacked_value = torch.stack(
                            [value] * fake_next_action.shape[1], axis=1
                        )
                        new_next_state.set(key, stacked_value)
            else:
                new_next_state = torch.stack(
                    [next_state] * fake_next_action.shape[1], axis=1
                )
            next_energy = (
                self.q_target(
                    fake_next_action,
                    new_next_state,
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
        return q_loss


class QGPOPolicy(nn.Module):
    """
    Overview:
        QGPO policy network.
    Interfaces:
        ``__init__``, ``forward``, ``sample``, ``behaviour_policy_sample``, ``compute_q``, ``behaviour_policy_loss``, ``energy_guidance_loss``, ``q_loss``
    """

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device

        self.critic = QGPOCritic(config.critic)
        self.diffusion_model = EnergyConditionalDiffusionModel(
            config.diffusion_model, energy_model=self.critic
        )

    def forward(
        self, state: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
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
        guidance_scale: Union[torch.Tensor, float] = torch.tensor(1.0),
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of QGPO policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            guidance_scale (:obj:`Union[torch.Tensor, float]`): The guidance scale.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            with_grad=False,
            solver_config=solver_config,
        )

    def behaviour_policy_sample(
        self,
        state: Union[torch.Tensor, TensorDict],
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        solver_config: EasyDict = None,
        t_span: torch.Tensor = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of behaviour policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            solver_config (:obj:`EasyDict`): The configuration for the ODE solver.
            t_span (:obj:`torch.Tensor`): The time span for the ODE solver or SDE solver.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.diffusion_model.sample_without_energy_guidance(
            t_span=t_span,
            condition=state,
            batch_size=batch_size,
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
    ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            action (:obj:`torch.Tensor`): The input action.
            state (:obj:`torch.Tensor`): The input state.
        """

        return self.diffusion_model.score_matching_loss(
            action, state, weighting_scheme="vanilla"
        )

    def energy_guidance_loss(
        self,
        state: Union[torch.Tensor, TensorDict],
        fake_next_action: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the energy guidance loss of QGPO.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            fake_next_action (:obj:`Union[torch.Tensor, TensorDict]`): The input fake next action.
        """

        return self.diffusion_model.energy_guidance_loss(fake_next_action, state)

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


class QGPOAlgorithm:
    """
    Overview:
        Q-guided policy optimization (QGPO) algorithm, which is an offline reinforcement learning algorithm that uses energy-based diffusion model for policy modeling.
    Interfaces:
        ``__init__``, ``train``, ``deploy``
    """

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: QGPODataset = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the QGPO algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            dataset (:obj:`QGPODataset`): The dataset.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator
        self.dataset = dataset

        # ---------------------------------------
        # Customized model initialization code ↓
        # ---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        if model is not None:
            self.model = model
            self.behaviour_policy_train_epoch = 0
            self.energy_guidance_train_epoch = 0
            self.critic_train_epoch = 0
        else:
            self.model = torch.nn.ModuleDict()
            config = self.config.train
            assert hasattr(config.model, "QGPOPolicy")

            if torch.__version__ >= "2.0.0":
                self.model["QGPOPolicy"] = torch.compile(
                    QGPOPolicy(config.model.QGPOPolicy).to(
                        config.model.QGPOPolicy.device
                    )
                )
            else:
                self.model["QGPOPolicy"] = QGPOPolicy(config.model.QGPOPolicy).to(
                    config.model.QGPOPolicy.device
                )

            if (
                hasattr(config.parameter, "checkpoint_path")
                and config.parameter.checkpoint_path is not None
            ):
                self.behaviour_policy_train_epoch = load_model(
                    path=config.parameter.checkpoint_path,
                    model=self.model["QGPOPolicy"].diffusion_model.model,
                    optimizer=None,
                    prefix="behaviour_policy",
                )

                self.energy_guidance_train_epoch = load_model(
                    path=config.parameter.checkpoint_path,
                    model=self.model["QGPOPolicy"].diffusion_model.energy_guidance,
                    optimizer=None,
                    prefix="energy_guidance",
                )

                self.critic_train_epoch = load_model(
                    path=config.parameter.checkpoint_path,
                    model=self.model["QGPOPolicy"].critic,
                    optimizer=None,
                    prefix="critic",
                )
            else:
                self.behaviour_policy_train_epoch = 0
                self.energy_guidance_train_epoch = 0
                self.critic_train_epoch = 0

        # ---------------------------------------
        # Customized model initialization code ↑
        # ---------------------------------------

    def train(self, config: EasyDict = None):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """

        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )

        with wandb.init(**config.wandb) as wandb_run:
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

            def generate_fake_action(model, states, action_augment_num):
                # model.eval()
                fake_actions_sampled = []
                if isinstance(states, TensorDict):
                    from torchrl.data import LazyTensorStorage

                    storage = LazyTensorStorage(max_size=states.shape[0])
                    storage.set(
                        range(states.shape[0]),
                        TensorDict(
                            {
                                "s": states,
                            },
                            batch_size=[states.shape[0]],
                        ),
                    )
                    for index in torch.split(torch.arange(0, states.shape[0], 1), 4096):
                        index = index.int()
                        data = storage[index]
                        fake_actions_per_state = []
                        for _ in range(action_augment_num):
                            fake_actions_per_state.append(
                                model.sample(
                                    state=data["s"].to(config.model.QGPOPolicy.device),
                                    guidance_scale=0.0,
                                    t_span=(
                                        torch.linspace(
                                            0.0, 1.0, config.parameter.fake_data_t_span
                                        ).to(config.model.QGPOPolicy.device)
                                        if config.parameter.fake_data_t_span is not None
                                        else None
                                    ),
                                )
                            )
                        fake_actions_sampled.append(
                            torch.stack(fake_actions_per_state, dim=1)
                        )
                else:
                    for states in track(
                        np.array_split(states, states.shape[0] // 4096 + 1),
                        description="Generate fake actions",
                    ):
                        # TODO: mkae it batchsize
                        fake_actions_per_state = []
                        for _ in range(action_augment_num):
                            fake_actions_per_state.append(
                                model.sample(
                                    state=states,
                                    guidance_scale=0.0,
                                    t_span=(
                                        torch.linspace(
                                            0.0, 1.0, config.parameter.fake_data_t_span
                                        ).to(states.device)
                                        if config.parameter.fake_data_t_span is not None
                                        else None
                                    ),
                                )
                            )
                        fake_actions_sampled.append(
                            torch.stack(fake_actions_per_state, dim=1)
                        )
                fake_actions = torch.cat(fake_actions_sampled, dim=0)
                return fake_actions

            def evaluate(model, epoch):
                evaluation_results = dict()
                for guidance_scale in config.parameter.evaluation.guidance_scale:

                    def policy(obs: np.ndarray) -> np.ndarray:
                        if isinstance(obs, np.ndarray):
                            obs = torch.tensor(
                                obs,
                                dtype=torch.float32,
                                device=config.model.QGPOPolicy.device,
                            ).unsqueeze(0)
                        elif isinstance(obs, dict):
                            for key in obs:
                                obs[key] = torch.tensor(
                                    obs[key],
                                    dtype=torch.float32,
                                    device=config.model.QGPOPolicy.device,
                                ).unsqueeze(0)
                                if obs[key].dim() == 1 and obs[key].shape[0] == 1:
                                    obs[key] = obs[key].unsqueeze(1)
                            obs = TensorDict(obs, batch_size=[1])
                        else:
                            raise ValueError("Unsupported observation type.")
                        action = (
                            model.sample(
                                state=obs,
                                guidance_scale=guidance_scale,
                                t_span=(
                                    torch.linspace(
                                        0.0, 1.0, config.parameter.fake_data_t_span
                                    ).to(config.model.QGPOPolicy.device)
                                    if config.parameter.fake_data_t_span is not None
                                    else None
                                ),
                            )
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        return action

                    evaluation_results[
                        f"evaluation/guidance_scale:[{guidance_scale}]/total_return"
                    ] = self.simulator.evaluate(policy=policy,)[0]["total_return"]
                    log.info(
                        f"Train epoch: {epoch}, guidance_scale: {guidance_scale}, total_return: {evaluation_results[f'evaluation/guidance_scale:[{guidance_scale}]/total_return']}"
                    )

                return evaluation_results

            replay_buffer = TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.behaviour_policy.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for epoch in track(
                range(config.parameter.behaviour_policy.epochs),
                description="Behaviour policy training",
            ):

                if self.behaviour_policy_train_epoch >= epoch:
                    continue

                counter = 0
                behaviour_model_training_loss_sum = 0.0
                for index, data in enumerate(replay_buffer):

                    behaviour_model_training_loss = self.model[
                        "QGPOPolicy"
                    ].behaviour_policy_loss(
                        data["a"].to(config.model.QGPOPolicy.device),
                        data["s"].to(config.model.QGPOPolicy.device),
                    )
                    behaviour_model_optimizer.zero_grad()
                    behaviour_model_training_loss.backward()
                    behaviour_model_optimizer.step()

                    counter += 1
                    behaviour_model_training_loss_sum += (
                        behaviour_model_training_loss.item()
                    )

                if (
                    epoch == 0
                    or (epoch + 1) % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(self.model["QGPOPolicy"], epoch=epoch)
                    wandb_run.log(data=evaluation_results, commit=False)
                    save_model(
                        path=config.parameter.checkpoint_path,
                        model=self.model["QGPOPolicy"].diffusion_model.model,
                        optimizer=behaviour_model_optimizer,
                        iteration=epoch,
                        prefix="behaviour_policy",
                    )

                self.behaviour_policy_train_epoch = epoch

                wandb_run.log(
                    data=dict(
                        behaviour_policy_train_epoch=epoch,
                        behaviour_model_training_loss=behaviour_model_training_loss_sum
                        / counter,
                    ),
                    commit=True,
                )

            fake_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.states[:].to(config.model.QGPOPolicy.device),
                config.parameter.action_augment_num,
            ).to("cpu")
            fake_next_actions = generate_fake_action(
                self.model["QGPOPolicy"],
                self.dataset.next_states[:].to(config.model.QGPOPolicy.device),
                config.parameter.action_augment_num,
            ).to("cpu")

            self.dataset.load_fake_actions(
                fake_actions=fake_actions,
                fake_next_actions=fake_next_actions,
            )

            # TODO add notation
            replay_buffer = TensorDictReplayBuffer(
                storage=self.dataset.storage,
                batch_size=config.parameter.energy_guided_policy.batch_size,
                sampler=SamplerWithoutReplacement(),
                prefetch=10,
                pin_memory=True,
            )

            q_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].critic.q.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            energy_guidance_optimizer = torch.optim.Adam(
                self.model["QGPOPolicy"].diffusion_model.energy_guidance.parameters(),
                lr=config.parameter.energy_guidance.learning_rate,
            )

            with Progress() as progress:
                critic_training = progress.add_task(
                    "Critic training",
                    total=config.parameter.critic.stop_training_epochs,
                )
                energy_guidance_training = progress.add_task(
                    "Energy guidance training",
                    total=config.parameter.energy_guidance.epochs,
                )

                for epoch in range(config.parameter.energy_guidance.epochs):

                    if self.energy_guidance_train_epoch >= epoch:
                        continue

                    counter = 0
                    q_loss_sum = 0.0
                    energy_guidance_loss_sum = 0.0

                    for index, data in enumerate(replay_buffer):

                        if epoch < config.parameter.critic.stop_training_epochs:

                            q_loss = self.model["QGPOPolicy"].q_loss(
                                data["a"].to(config.model.QGPOPolicy.device),
                                data["s"].to(config.model.QGPOPolicy.device),
                                data["r"].to(config.model.QGPOPolicy.device),
                                data["s_"].to(config.model.QGPOPolicy.device),
                                data["d"].to(config.model.QGPOPolicy.device),
                                data["fake_a_"].to(config.model.QGPOPolicy.device),
                                discount_factor=config.parameter.critic.discount_factor,
                            )

                            q_optimizer.zero_grad()
                            q_loss.backward()
                            q_optimizer.step()
                            q_loss_sum += q_loss.item()

                            # Update target
                            for param, target_param in zip(
                                self.model["QGPOPolicy"].critic.q.parameters(),
                                self.model["QGPOPolicy"].critic.q_target.parameters(),
                            ):
                                target_param.data.copy_(
                                    config.parameter.critic.update_momentum * param.data
                                    + (1 - config.parameter.critic.update_momentum)
                                    * target_param.data
                                )

                        energy_guidance_loss = self.model[
                            "QGPOPolicy"
                        ].energy_guidance_loss(
                            data["s"].to(config.model.QGPOPolicy.device),
                            data["fake_a"].to(config.model.QGPOPolicy.device),
                        )
                        energy_guidance_optimizer.zero_grad()
                        energy_guidance_loss.backward()
                        energy_guidance_optimizer.step()
                        energy_guidance_loss_sum += energy_guidance_loss.item()

                        counter += 1

                    if epoch < config.parameter.critic.stop_training_epochs:
                        progress.update(critic_training, advance=1)
                    progress.update(energy_guidance_training, advance=1)

                    if (
                        epoch == 0
                        or (epoch + 1) % config.parameter.evaluation.evaluation_interval
                        == 0
                    ):
                        evaluation_results = evaluate(
                            self.model["QGPOPolicy"], epoch=epoch
                        )
                        wandb_run.log(data=evaluation_results, commit=False)
                        save_model(
                            path=config.parameter.checkpoint_path,
                            model=self.model[
                                "QGPOPolicy"
                            ].diffusion_model.energy_guidance,
                            optimizer=energy_guidance_optimizer,
                            iteration=epoch,
                            prefix="energy_guidance",
                        )
                        save_model(
                            path=config.parameter.checkpoint_path,
                            model=self.model["QGPOPolicy"].critic,
                            optimizer=q_optimizer,
                            iteration=epoch,
                            prefix="critic",
                        )

                    self.energy_guidance_train_epoch = epoch
                    self.critic_train_epoch = epoch

                    wandb_run.log(
                        data=dict(
                            energy_guidance_train_epoch=epoch,
                            critic_train_epoch=epoch,
                            q_loss=q_loss_sum / counter,
                            energy_guidance_loss=energy_guidance_loss_sum / counter,
                        ),
                        commit=True,
                    )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

            wandb.finish()

    def deploy(self, config: EasyDict = None) -> QGPOAgent:
        """
        Overview:
            Deploy the model using the given configuration.
        Arguments:
            config (:obj:`EasyDict`): The deployment configuration.
        """

        if config is not None:
            config = merge_two_dicts_into_newone(self.config.deploy, config)
        else:
            config = self.config.deploy

        assert "QGPOPolicy" in self.model, "The model must be trained first."
        return QGPOAgent(
            config=config,
            model=copy.deepcopy(self.model),
        )
