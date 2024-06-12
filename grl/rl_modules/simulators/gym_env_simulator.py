from typing import Callable, Dict, List, Tuple, Union

import gym
import torch
from easydict import EasyDict


class GymEnvSimulator:
    """
    Overview:
        A simple gym environment simulator in GenerativeRL.
        This simulator is used to collect episodes and steps using a given policy in a gym environment.
        It runs in single process and is suitable for small-scale experiments.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_steps``, ``evaluate``
    """

    def __init__(self, env_id: str) -> None:
        """
        Overview:
            Initialize the GymEnvSimulator according to the given configuration.
        Arguments:
            env_id (:obj:`str`): The id of the gym environment to simulate.
        """
        self.env_id = env_id
        self.collect_env = gym.make(self.env_id)

        if gym.__version__ >= "0.26.0":
            self.last_state_obs, _ = self.collect_env.reset()
            self.last_state_done = False
            self.last_state_truncated = False
        else:
            self.last_state_obs = self.collect_env.reset()
            self.last_state_done = False

        self.observation_space = self.collect_env.observation_space
        self.action_space = self.collect_env.action_space

    def collect_episodes(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
        num_steps: int = None,
    ) -> List[Dict]:
        """
        Overview:
            Collect several episodes using the given policy. The environment will be reset at the beginning of each episode.
            No history will be stored in this method. The collected information of steps will be returned as a list of dictionaries.
        Arguments:
            policy (:obj:`Union[Callable, torch.nn.Module]`): The policy to collect episodes.
            num_episodes (:obj:`int`): The number of episodes to collect.
            num_steps (:obj:`int`): The number of steps to collect.
        """
        assert num_episodes is not None or num_steps is not None
        if num_episodes is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= "0.26.0":
                    for i in range(num_episodes):
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done and not truncated:
                            action = policy(obs)
                            next_obs, reward, done, truncated, _ = (
                                self.collect_env.step(action)
                            )
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    truncated=truncated,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
                else:
                    for i in range(num_episodes):
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            action = policy(obs)
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= "0.26.0":
                    while len(data_list) < num_steps:
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done and not truncated:
                            action = policy(obs)
                            next_obs, reward, done, truncated, _ = (
                                self.collect_env.step(action)
                            )
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    truncated=truncated,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
                else:
                    while len(data_list) < num_steps:
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            action = policy(obs)
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
            return data_list

    def collect_steps(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
        num_steps: int = None,
        random_policy: bool = False,
    ) -> List[Dict]:
        """
        Overview:
            Collect several steps using the given policy. The environment will not be reset until the end of the episode.
            Last observation will be stored in this method. The collected information of steps will be returned as a list of dictionaries.
        Arguments:
            policy (:obj:`Union[Callable, torch.nn.Module]`): The policy to collect steps.
            num_episodes (:obj:`int`): The number of episodes to collect.
            num_steps (:obj:`int`): The number of steps to collect.
            random_policy (:obj:`bool`): Whether to use a random policy.
        """
        assert num_episodes is not None or num_steps is not None
        if num_episodes is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= "0.26.0":
                    for i in range(num_episodes):
                        obs, _ = self.collect_env.reset()
                        done = False
                        truncated = False
                        while not done and not truncated:
                            if random_policy:
                                action = self.collect_env.action_space.sample()
                            else:
                                action = policy(obs)
                            next_obs, reward, done, truncated, _ = (
                                self.collect_env.step(action)
                            )
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    truncated=truncated,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
                    self.last_state_obs, _ = self.collect_env.reset()
                    self.last_state_done = False
                    self.last_state_truncated = False
                else:
                    for i in range(num_episodes):
                        obs = self.collect_env.reset()
                        done = False
                        while not done:
                            if random_policy:
                                action = self.collect_env.action_space.sample()
                            else:
                                action = policy(obs)
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append(
                                dict(
                                    obs=obs,
                                    action=action,
                                    reward=reward,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            obs = next_obs
                    self.last_state_obs = self.collect_env.reset()
                    self.last_state_done = False
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= "0.26.0":
                    while len(data_list) < num_steps:
                        if not self.last_state_done or not self.last_state_truncated:
                            if random_policy:
                                action = self.collect_env.action_space.sample()
                            else:
                                action = policy(self.last_state_obs)
                            next_obs, reward, done, truncated, _ = (
                                self.collect_env.step(action)
                            )
                            data_list.append(
                                dict(
                                    obs=self.last_state_obs,
                                    action=action,
                                    reward=reward,
                                    truncated=truncated,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            self.last_state_obs = next_obs
                            self.last_state_done = done
                            self.last_state_truncated = truncated
                        else:
                            self.last_state_obs, _ = self.collect_env.reset()
                            self.last_state_done = False
                            self.last_state_truncated = False
                else:
                    while len(data_list) < num_steps:
                        if not self.last_state_done:
                            if random_policy:
                                action = self.collect_env.action_space.sample()
                            else:
                                action = policy(self.last_state_obs)
                            next_obs, reward, done, _ = self.collect_env.step(action)
                            data_list.append(
                                dict(
                                    obs=self.last_state_obs,
                                    action=action,
                                    reward=reward,
                                    done=done,
                                    next_obs=next_obs,
                                )
                            )
                            self.last_state_obs = next_obs
                            self.last_state_done = done
                        else:
                            self.last_state_obs = self.collect_env.reset()
                            self.last_state_done = False
            return data_list

    def evaluate(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
        render_args: Dict = None,
    ) -> List[Dict]:
        """
        Overview:
            Evaluate the given policy using the environment. The environment will be reset at the beginning of each episode.
            No history will be stored in this method. The evaluation resultswill be returned as a list of dictionaries.
        """
        if num_episodes is None:
            num_episodes = 1

        if render_args is not None:
            render = True
        else:
            render = False

        def render_env(env, render_args):
            # TODO: support different render modes
            render_output = env.render(
                **render_args,
            )
            return render_output

        eval_results = []

        env = gym.make(self.env_id)
        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                if gym.__version__ >= "0.26.0":
                    obs, _ = env.reset()
                    if render:
                        render_output.append(render_env(env, render_args))
                    done = False
                    truncated = False
                    while not done and not truncated:
                        action = policy(obs)
                        next_obs, reward, done, truncated, _ = env.step(action)
                        data_list.append(
                            dict(
                                obs=obs,
                                action=action,
                                reward=reward,
                                truncated=truncated,
                                done=done,
                                next_obs=next_obs,
                            )
                        )
                        obs = next_obs
                    if render:
                        render_output.append(render_env(env, render_args))
                else:
                    step = 0
                    obs = env.reset()
                    if render:
                        render_output.append(render_env(env, render_args))
                    done = False
                    while not done:
                        action = policy(obs)
                        next_obs, reward, done, _ = env.step(action)
                        step += 1
                        if render:
                            render_output.append(render_env(env, render_args))
                        data_list.append(
                            dict(
                                obs=obs,
                                action=action,
                                reward=reward,
                                done=done,
                                next_obs=next_obs,
                            )
                        )
                        obs = next_obs
                    if render:
                        render_output.append(render_env(env, render_args))

            eval_results.append(
                dict(
                    total_return=sum([d["reward"] for d in data_list]),
                    total_steps=len(data_list),
                    data_list=data_list,
                    render_output=render_output if render else None,
                )
            )

        return eval_results
