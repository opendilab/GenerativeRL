import copy
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.utils.data import DataLoader

import wandb
from grl.agents.srpo import SRPOAgent
from grl.datasets import create_dataset
from grl.datasets.d4rl import D4RLDataset
from grl.generative_models.sro import SRPOConditionalDiffusionModel
from grl.neural_network import MultiLayerPerceptron
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network.q_network import DoubleQNetwork
from grl.utils import set_seed
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


class Dirac_Policy(nn.Module):
    """
    Overview:
        The deterministic policy network used in SRPO algorithm.
    Interfaces:
        ``__init__``, ``forward``, ``select_actions``
    """

    def __init__(self, action_dim: int, state_dim: int, layer: int = 2):
        super().__init__()
        self.net = MultiLayerPerceptron(
            hidden_sizes=[state_dim] + [256 for _ in range(layer)],
            output_size=action_dim,
            activation="relu",
            final_activation="tanh",
        )

    def forward(self, state: torch.Tensor):
        return self.net(state)

    def select_actions(self, state: torch.Tensor):
        return self(state)


def asymmetric_l2_loss(u, tau):
    """
    Overview:
        Calculate the asymmetric L2 loss, which is used in Implicit Q-Learning.
    Arguments:
        u (:obj:`torch.Tensor`): The input tensor.
        tau (:obj:`float`): The threshold.
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ValueFunction(nn.Module):
    """
    Overview:
        The value network used in SRPO algorithm.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, state_dim: int):
        """
        Overview:
            Initialize the value network.
        Arguments:
            state_dim (:obj:`int`): The dimension of the state.
        """
        super().__init__()
        self.v = MultiLayerPerceptron(
            hidden_sizes=[state_dim, 256, 256],
            output_size=1,
            activation="relu",
        )

    def forward(self, state):
        """
        Overview:
            Forward pass of the value network.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
        """
        return self.v(state)


class SRPOCritic(nn.Module):
    """
    Overview:
        The critic network used in SRPO algorithm.
    Interfaces:
        ``__init__``, ``v_loss``, ``q_loss
    """

    def __init__(self, config) -> None:
        """
        Overview:
            Initialize the critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()
        self.q0 = DoubleQNetwork(config.DoubleQNetwork)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False)
        self.vf = ValueFunction(config.sdim)

    def v_loss(self, state, action, next_state, tau):
        """
        Overview:
            Calculate the value loss.
        Arguments:
            data (:obj:`Dict`): The input data.
            tau (:obj:`float`): The threshold.
        """

        with torch.no_grad():
            target_q = self.q0_target(action, state).detach()
            next_v = self.vf(next_state).detach()
        # Update value function
        v = self.vf(state)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, tau)
        return v_loss, next_v

    def q_loss(self, state, action, reward, done, next_v, discount):
        """
        Overview:
            Calculate the Q loss.
        Arguments:
            data (:obj:`Dict`): The input data.
            next_v (:obj:`torch.Tensor`): The input next state value.
            discount (:obj:`float`): The discount factor.
        """
        # Update Q function
        targets = reward + (1.0 - done.float()) * discount * next_v.detach()
        qs = self.q0.compute_double_q(action, state)
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        return q_loss


class SRPOPolicy(nn.Module):
    """
    Overview:
        The SRPO policy network.
    Interfaces:
        ``__init__``, ``forward``, ``behaviour_policy_loss``, ``v_loss``, ``q_loss``, ``srpo_actor_loss``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the SRPO policy network.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()
        self.config = config
        self.device = config.device

        self.deter_policy = Dirac_Policy(**config.policy_model)
        self.critic = SRPOCritic(config.critic)
        self.sro = SRPOConditionalDiffusionModel(
            config=config.diffusion_model,
            value_model=self.critic,
            distribution_model=self.deter_policy,
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
        return self.deter_policy.select_actions(state)

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

        return self.sro.score_matching_loss(action, state)

    def v_loss(
        self,
        state, action, next_state,
        tau: int = 0.9,
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
        """
        v_loss, next_v = self.critic.v_loss(state, action, next_state, tau)
        return v_loss, next_v

    def q_loss(
        self,
        state,
        action,
        reward,
        done,
        next_v: torch.Tensor,
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

        loss = self.critic.q_loss(state, action, reward, done, next_v, discount_factor)
        return loss

    def srpo_actor_loss(
        self,
        state,
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
        loss, q = self.sro.srpo_loss(state)
        return loss, q


class SRPOAlgorithm:

    def __init__(
        self,
        config: EasyDict = None,
        simulator=None,
        dataset: D4RLDataset = None,
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
        set_seed(self.config.deploy.env["seed"])

        config = (
            merge_two_dicts_into_newone(
                self.config.train if hasattr(self.config, "train") else EasyDict(),
                config,
            )
            if config is not None
            else self.config.train
        )

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        directory_path = os.path.join(
            f"./{config.project}",
            formatted_time,
        )
        os.makedirs(directory_path, exist_ok=True)
        with wandb.init(
            project=(
                config.project if hasattr(config, "project") else __class__.__name__
            ),
            **config.wandb if hasattr(config, "wandb") else {},
        ) as wandb_run:
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
            # Customized model initialization code ↓
            # ---------------------------------------
            if hasattr(config.model, "SRPOPolicy"):
                self.model["SRPOPolicy"] = SRPOPolicy(config.model.SRPOPolicy)
                self.model["SRPOPolicy"].to(config.model.SRPOPolicy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["SRPOPolicy"] = torch.compile(self.model["SRPOPolicy"])
            # ---------------------------------------
            # test model ↓
            # ---------------------------------------
            assert isinstance(
                self.model, (torch.nn.Module, torch.nn.ModuleDict)
            ), "self.model must be torch.nn.Module or torch.nn.ModuleDict."
            if isinstance(self.model, torch.nn.ModuleDict):
                assert (
                    "SRPOPolicy" in self.model and self.model["SRPOPolicy"]
                ), "self.model['SRPOPolicy'] cannot be empty."
            else:  # self.model is torch.nn.Module
                assert self.model, "self.model cannot be empty."
            # ---------------------------------------
            # Customized model initialization code ↑
            # ---------------------------------------

            # ---------------------------------------
            # Customized training code ↓
            # ---------------------------------------

            def get_train_data(dataloader):
                while True:
                    yield from dataloader

            def pallaral_simple_eval_policy(
                policy_fn, env_name, seed, eval_episodes=20
            ):
                eval_envs = []
                for i in range(eval_episodes):
                    env = gym.make(env_name)
                    eval_envs.append(env)
                    env.seed(seed + 1001 + i)
                    env.buffer_state = env.reset()
                    env.buffer_return = 0.0
                ori_eval_envs = [env for env in eval_envs]
                import time

                t = time.time()
                while len(eval_envs) > 0:
                    new_eval_envs = []
                    states = np.stack([env.buffer_state for env in eval_envs])
                    states = torch.Tensor(states).to(config.model.SRPOPolicy.device)
                    with torch.no_grad():
                        actions = policy_fn(states).detach().cpu().numpy()
                    for i, env in enumerate(eval_envs):
                        state, reward, done, info = env.step(actions[i])
                        env.buffer_return += reward
                        env.buffer_state = state
                        if not done:
                            new_eval_envs.append(env)
                    eval_envs = new_eval_envs
                for i in range(eval_episodes):
                    ori_eval_envs[i].buffer_return = d4rl.get_normalized_score(
                        env_name, ori_eval_envs[i].buffer_return
                    )
                mean = np.mean(
                    [ori_eval_envs[i].buffer_return for i in range(eval_episodes)]
                )
                std = np.std(
                    [ori_eval_envs[i].buffer_return for i in range(eval_episodes)]
                )
                return mean, std

            def evaluate(policy_fn, train_iter):
                evaluation_results = dict()

                def policy(obs: np.ndarray) -> np.ndarray:
                    obs = torch.tensor(
                        obs, dtype=torch.float32, device=config.model.SRPOPolicy.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_fn(obs).squeeze(0).detach().cpu().numpy()
                    return action

                result = self.simulator.evaluate(
                    policy=policy,
                )[0]
                evaluation_results["evaluation/total_return"] = result["total_return"]
                evaluation_results["evaluation/total_steps"] = result["total_steps"]
                return evaluation_results

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.behaviour_policy.batch_size,
                    shuffle=True,
                    collate_fn=None,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=8,
                )
            )

            behaviour_model_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].sro.diffusion_model.model.parameters(),
                lr=config.parameter.behaviour_policy.learning_rate,
            )

            for train_diffusion_iter in track(
                range(config.parameter.behaviour_policy.iterations),
                description="Behaviour policy training",
            ):
                data = next(data_generator)
                behaviour_model_training_loss = self.model[
                    "SRPOPolicy"
                ].behaviour_policy_loss(data["a"].to(config.model.SRPOPolicy.device), data["s"].to(config.model.SRPOPolicy.device))
                behaviour_model_optimizer.zero_grad()
                behaviour_model_training_loss.backward()
                behaviour_model_optimizer.step()

                wandb_run.log(
                    data=dict(
                        train_diffusion_iter=train_diffusion_iter,
                        behaviour_model_training_loss=behaviour_model_training_loss.item(),
                    ),
                    commit=True,
                )

            if train_diffusion_iter == config.parameter.behaviour_policy.iterations - 1:
                file_path = os.path.join(
                    directory_path, f"checkpoint_diffusion_{train_diffusion_iter+1}.pt"
                )
                torch.save(
                    dict(
                        diffusion_model=self.model[
                            "SRPOPolicy"
                        ].sro.diffusion_model.model.state_dict(),
                        behaviour_model_optimizer=behaviour_model_optimizer.state_dict(),
                        diffusion_iteration=train_diffusion_iter + 1,
                    ),
                    f=file_path,
                )

            q_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].critic.q0.parameters(),
                lr=config.parameter.critic.learning_rate,
            )
            v_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].critic.vf.parameters(),
                lr=config.parameter.critic.learning_rate,
            )

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.critic.batch_size,
                    shuffle=True,
                    collate_fn=None,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=8,
                )
            )

            for train_critic_iter in track(
                range(config.parameter.critic.iterations), description="Critic training"
            ):
                data = next(data_generator)

                v_loss, next_v = self.model["SRPOPolicy"].v_loss(
                    state=data["s"].to(config.model.SRPOPolicy.device),
                    action=data["a"].to(config.model.SRPOPolicy.device),
                    next_state=data["s_"].to(config.model.SRPOPolicy.device),
                    tau=config.parameter.critic.tau,
                )
                v_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                v_optimizer.step()

                q_loss = self.model["SRPOPolicy"].q_loss(
                    state=data["s"].to(config.model.SRPOPolicy.device),
                    action=data["a"].to(config.model.SRPOPolicy.device),
                    reward=data["r"].to(config.model.SRPOPolicy.device),
                    done=data["d"].to(config.model.SRPOPolicy.device),
                    next_v=next_v,
                    discount=config.parameter.critic.discount_factor,
                )
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                q_optimizer.step()

                # Update target
                for param, target_param in zip(
                    self.model["SRPOPolicy"].critic.q0.parameters(),
                    self.model["SRPOPolicy"].critic.q0_target.parameters(),
                ):
                    target_param.data.copy_(
                        config.parameter.critic.moment * param.data
                        + (1 - config.parameter.critic.moment) * target_param.data
                    )

                wandb_run.log(
                    data=dict(
                        train_critic_iter=train_critic_iter,
                        q_loss=q_loss.item(),
                        v_loss=v_loss.item(),
                    ),
                    commit=True,
                )

            if train_critic_iter == config.parameter.critic.iterations - 1:
                file_path = os.path.join(
                    directory_path, f"checkpoint_critic_{train_critic_iter+1}.pt"
                )
                torch.save(
                    dict(
                        q_model=self.model["SRPOPolicy"].critic.q0.state_dict(),
                        v_model=self.model["SRPOPolicy"].critic.vf.state_dict(),
                        q_optimizer=q_optimizer.state_dict(),
                        v_optimizer=v_optimizer.state_dict(),
                        critic_iteration=train_critic_iter + 1,
                    ),
                    f=file_path,
                )

            data_generator = get_train_data(
                DataLoader(
                    self.dataset,
                    batch_size=config.parameter.actor.batch_size,
                    shuffle=True,
                    collate_fn=None,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=8,
                )
            )
            SRPO_policy_optimizer = torch.optim.Adam(
                self.model["SRPOPolicy"].deter_policy.parameters(), lr=3e-4
            )
            SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                SRPO_policy_optimizer,
                T_max=config.parameter.actor.iterations,
                eta_min=0.0,
            )
            for train_policy_iter in track(
                range(config.parameter.actor.iterations), description="actor training"
            ):
                data = next(data_generator)
                self.model["SRPOPolicy"].sro.diffusion_model.model.eval()
                actor_loss, q = self.model["SRPOPolicy"].srpo_actor_loss(data["s"].to(config.model.SRPOPolicy.device))
                actor_loss = actor_loss.sum(-1).mean()
                SRPO_policy_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                SRPO_policy_optimizer.step()
                SRPO_policy_lr_scheduler.step()
                self.model["SRPOPolicy"].sro.diffusion_model.model.train()
                wandb_run.log(
                    data=dict(
                        train_policy_iter=train_policy_iter,
                        actor_loss=actor_loss,
                        q=q,
                    ),
                    commit=True,
                )

                if (
                    train_policy_iter == 0
                    or (train_policy_iter + 1)
                    % config.parameter.evaluation.evaluation_interval
                    == 0
                ):
                    evaluation_results = evaluate(
                        self.model["SRPOPolicy"], train_iter=train_policy_iter
                    )

                    wandb_run.log(
                        data=evaluation_results,
                        commit=False,
                    )

                if train_policy_iter == config.parameter.actor.iterations - 1:
                    file_path = os.path.join(
                        directory_path, f"checkpoint_policy_{train_policy_iter+1}.pt"
                    )
                    torch.save(
                        dict(
                            actor_model=self.model[
                                "SRPOPolicy"
                            ].deter_policy.state_dict(),
                            actor_optimizer=SRPO_policy_optimizer.state_dict(),
                            policy_iteration=train_policy_iter + 1,
                        ),
                        f=file_path,
                    )

            # ---------------------------------------
            # Customized training code ↑
            # ---------------------------------------

        wandb.finish()

    def deploy(self, config: EasyDict = None) -> SRPOAgent:
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

        return SRPOAgent(
            config=config,
            model=torch.nn.ModuleDict(
                {
                    "SRPOPolicy": self.deter_policy.select_actions,
                }
            ),
        )
