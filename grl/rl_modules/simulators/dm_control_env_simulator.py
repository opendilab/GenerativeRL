from typing import Callable, Dict, List, Union

import numpy as np

if np.__version__ > "1.23.1":
    np.bool = np.bool_

import torch
import os
import gym


def partial_observation_rodent(obs_dict):
    # Define the keys you want to keep
    keys_to_keep = [
        "walker/joints_pos",
        "walker/joints_vel",
        "walker/tendons_pos",
        "walker/tendons_vel",
        "walker/appendages_pos",
        "walker/world_zaxis",
        "walker/sensors_accelerometer",
        "walker/sensors_velocimeter",
        "walker/sensors_gyro",
        "walker/sensors_touch",
        "walker/egocentric_camera",
    ]
    # Filter the observation dictionary to only include the specified keys
    filtered_obs = {key: obs_dict[key] for key in keys_to_keep if key in obs_dict}
    return filtered_obs


class DeepMindControlEnvSimulator:
    """
    Overview:
        DeepMind control environment simulator in GenerativeRL.
        Google DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.
        This simulator is used to collect episodes and steps using a given policy in a gym environment.
        It runs in single process and is suitable for small-scale experiments.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_steps``, ``evaluate``
    """

    def __init__(self, domain_name: str, task_name: str, dict_return=True) -> None:
        """
        Overview:
            Initialize the DeepMindControlEnvSimulator according to the given configuration.
        Arguments:
            domain_name (:obj:`str`): The domain name of the environment.
            task_name (:obj:`str`): The task name of the environment.
            dict_return (:obj:`bool`): Whether to return the observation as a dictionary.
        """
        if domain_name == "rodent" and task_name == "gaps":
            import os

            os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"  # we make it for 8 gpus
            from dm_control import composer
            from dm_control.locomotion.examples import basic_rodent_2020

            self.domain_name = domain_name
            self.task_name = task_name
            self.collect_env = basic_rodent_2020.rodent_run_gaps()
            self.action_space = self.collect_env.action_spec()
            self.partial_observation = True
            self.partial_observation_fn = partial_observation_rodent
        else:
            from dm_control import suite

            self.domain_name = domain_name
            self.task_name = task_name
            self.collect_env = suite.load(domain_name, task_name)
            self.action_space = self.collect_env.action_spec()
            self.partial_observation = False

        self.last_state_obs = self.collect_env.reset().observation
        self.last_state_done = False
        self.dict_return = dict_return

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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        if not self.dict_return:
                            obs_values = []
                            next_obs_values = []
                            for key, value in obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                obs_values.append(value)
                            for key, value in next_obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                next_obs_values.append(value)
                            obs = np.concatenate(obs_values, axis=0)
                            next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_steps):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        if not self.dict_return:
                            obs_values = []
                            next_obs_values = []
                            for key, value in obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                obs_values.append(value)
                            for key, value in next_obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                next_obs_values.append(value)
                            obs = np.concatenate(obs_values, axis=0)
                            next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        if random_policy:
                            action = np.random.uniform(
                                self.action_space.minimum,
                                self.action_space.maximum,
                                size=self.action_space.shape,
                            )
                        else:
                            action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        if not self.dict_return:
                            obs_values = []
                            next_obs_values = []
                            for key, value in self.last_state_obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                obs_values.append(value)
                            for key, value in next_obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                next_obs_values.append(value)
                            obs_flatten = np.concatenate(obs_values, axis=0)
                            next_obs_flatten = np.concatenate(next_obs_values, axis=0)
                        data_list.append(
                            dict(
                                obs=obs_flatten,
                                action=action,
                                reward=reward,
                                done=done,
                                next_obs=next_obs_flatten,
                            )
                        )
                        obs = next_obs
                self.last_state_obs = self.collect_env.reset().observation
                self.last_state_done = False
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                for i in range(num_steps):
                    if self.last_state_done:
                        self.last_state_obs = self.collect_env.reset().observation
                        self.last_state_done = False
                    if random_policy:
                        action = np.random.uniform(
                            self.action_space.minimum,
                            self.action_space.maximum,
                            size=self.action_space.shape,
                        )
                    else:
                        if not self.dict_return:
                            obs_values = []
                            for key, value in self.last_state_obs.items():
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 3 and value.shape[0] == 1:
                                        value = value.reshape(1, -1)
                                elif np.isscalar(value):
                                    value = [value]
                                obs_values.append(value)
                            obs = np.concatenate(obs_values, axis=0)
                        action = policy(obs)
                    time_step = self.collect_env.step(action)
                    next_obs = time_step.observation
                    reward = time_step.reward
                    done = time_step.last()
                    if not self.dict_return:
                        obs_values = []
                        next_obs_values = []
                        for key, value in self.last_state_obs.items():
                            if isinstance(value, np.ndarray):
                                if value.ndim == 3 and value.shape[0] == 1:
                                    value = value.reshape(1, -1)
                            elif np.isscalar(value):
                                value = [value]
                            obs_values.append(value)
                        for key, value in next_obs.items():
                            if isinstance(value, np.ndarray):
                                if value.ndim == 3 and value.shape[0] == 1:
                                    value = value.reshape(1, -1)
                            elif np.isscalar(value):
                                value = [value]
                            next_obs_values.append(value)
                        obs_flatten = np.concatenate(obs_values, axis=0)
                        next_obs_flatten = np.concatenate(next_obs_values, axis=0)
                    data_list.append(
                        dict(
                            obs=obs_flatten,
                            action=action,
                            reward=reward,
                            done=done,
                            next_obs=next_obs_flatten,
                        )
                    )
                    self.last_state_obs = next_obs
                    self.last_state_done = done
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
        if "suite" not in globals():
            from dm_control import suite

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
        if self.domain_name == "rodent" and self.task_name == "gaps":
            import os

            os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
            from dm_control import composer
            from dm_control.locomotion.examples import basic_rodent_2020

            env = basic_rodent_2020.rodent_run_gaps()
        else:
            env = suite.load(self.domain_name, self.task_name)
        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                step = 0
                time_step = env.reset()
                obs = time_step.observation
                if render:
                    render_output.append(render_env(env, render_args))
                done = False
                action_spec = env.action_spec()
                while not done:
                    if self.partial_observation:
                        obs = self.partial_observation_fn(obs)
                    if not self.dict_return:
                        obs_values = []
                        for key, value in obs.items():
                            if isinstance(value, np.ndarray):
                                if value.ndim == 2:
                                    value = value.reshape(-1)
                            elif np.isscalar(value):
                                value = [value]
                            obs_values.append(value)
                        obs = np.concatenate(obs_values, axis=0)
                    action = policy(obs)
                    time_step = env.step(action)
                    next_obs = time_step.observation
                    reward = time_step.reward
                    done = time_step.last()
                    discount = time_step.discount
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
                            discount=discount,
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


class GymWrapper:

    def __init__(self, env, obs_key="image", act_key="action"):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._act_is_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs


class DeepMindControlVisualEnv:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, **kwargs):
        os.environ["MUJOCO_GL"] = "osmesa"  # 'egl'
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision")
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite

            self._env = suite.load(domain, task, **kwargs)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = ["orientations", "height", "velocity", "pixels"]
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs


class DeepMindControlVisualEnvSimulator:
    """
    Overview:
        DeepMind control environment simulator in GenerativeRL. This class differs from DeepMindControlEnvSimulator in that it is designed for visual observations.
        Google DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.
        This simulator is used to collect episodes and steps using a given policy in a gym environment.
        It runs in single process and is suitable for small-scale experiments.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_steps``, ``evaluate``
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
    ) -> None:
        """
        Overview:
            Initialize the DeepMindControlEnvSimulator according to the given configuration.
        Arguments:
            domain_name (:obj:`str`): The domain name of the environment.
            task_name (:obj:`str`): The task name of the environment.
            dict_return (:obj:`bool`): Whether to return the observation as a dictionary.
        """

        self.domain_name = domain_name
        self.task_name = task_name
        self.collect_env = DeepMindControlVisualEnv(
            name=f"{self.domain_name}_{self.task_name}"
        )
        self.observation_space = self.collect_env.obs_space
        self.action_space = self.collect_env.act_space

        self.last_state_obs = self.collect_env.reset()
        self.last_state_done = False

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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs = np.concatenate(obs_values, axis=0)
                        next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_steps):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs = np.concatenate(obs_values, axis=0)
                        next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        if random_policy:
                            action = np.random.uniform(
                                self.action_space.minimum,
                                self.action_space.maximum,
                                size=self.action_space.shape,
                            )
                        else:
                            action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs_flatten = np.concatenate(obs_values, axis=0)
                        next_obs_flatten = np.concatenate(next_obs_values, axis=0)
                        data_list.append(
                            dict(
                                obs=obs_flatten,
                                action=action,
                                reward=reward,
                                done=done,
                                next_obs=next_obs_flatten,
                            )
                        )
                        obs = next_obs
                self.last_state_obs = self.collect_env.reset().observation
                self.last_state_done = False
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                for i in range(num_steps):
                    if self.last_state_done:
                        self.last_state_obs = self.collect_env.reset().observation
                        self.last_state_done = False
                    if random_policy:
                        action = np.random.uniform(
                            self.action_space.minimum,
                            self.action_space.maximum,
                            size=self.action_space.shape,
                        )
                    else:
                        action = policy(obs)
                    time_step = self.collect_env.step(action)
                    next_obs = time_step.observation
                    reward = time_step.reward
                    done = time_step.last()
                    data_list.append(
                        dict(
                            obs=obs,
                            action=action,
                            reward=reward,
                            done=done,
                            next_obs=next_obs,
                        )
                    )
                    self.last_state_obs = next_obs
                    self.last_state_done = done
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

        env = DeepMindControlVisualEnv(name=f"{self.domain_name}_{self.task_name}")

        def render_env(env, render_args):
            # TODO: support different render modes
            render_output = env.render(
                **render_args,
            )
            return render_output

        eval_results = []

        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                step = 0
                time_step = env.reset()
                obs = time_step["image"]
                if render:
                    render_output.append(render_env(env, render_args))
                done = False

                while not done:

                    action = policy(obs)
                    time_step = env.step(action)
                    next_obs = time_step["image"]
                    reward = time_step["reward"]
                    done = time_step["is_last"] or time_step["is_terminal"]
                    discount = 1.0 - int(time_step["is_terminal"])
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
                            discount=discount,
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


class DeepMindControlVisualEnvSimulator2:
    """
    Overview:
        DeepMind control environment simulator in GenerativeRL. This class differs from DeepMindControlEnvSimulator in that it is designed for visual observations.
        Google DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.
        This simulator is used to collect episodes and steps using a given policy in a gym environment.
        It runs in single process and is suitable for small-scale experiments.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_steps``, ``evaluate``
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        stack_frames: int = 1,
    ) -> None:
        """
        Overview:
            Initialize the DeepMindControlEnvSimulator according to the given configuration.
        Arguments:
            domain_name (:obj:`str`): The domain name of the environment.
            task_name (:obj:`str`): The task name of the environment.
            dict_return (:obj:`bool`): Whether to return the observation as a dictionary.
        """

        self.domain_name = domain_name
        self.task_name = task_name
        self.collect_env = DeepMindControlVisualEnv(
            name=f"{self.domain_name}_{self.task_name}"
        )
        self.observation_space = self.collect_env.obs_space
        self.action_space = self.collect_env.act_space
        self.stack_frames = stack_frames

        self.last_state_obs = self.collect_env.reset()
        self.last_state_done = False

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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs = np.concatenate(obs_values, axis=0)
                        next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_steps):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs = np.concatenate(obs_values, axis=0)
                        next_obs = np.concatenate(next_obs_values, axis=0)
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
                for i in range(num_episodes):
                    obs = self.collect_env.reset().observation
                    done = False
                    while not done:
                        if random_policy:
                            action = np.random.uniform(
                                self.action_space.minimum,
                                self.action_space.maximum,
                                size=self.action_space.shape,
                            )
                        else:
                            action = policy(obs)
                        time_step = self.collect_env.step(action)
                        next_obs = time_step.observation
                        reward = time_step.reward
                        done = time_step.last()
                        obs_flatten = np.concatenate(obs_values, axis=0)
                        next_obs_flatten = np.concatenate(next_obs_values, axis=0)
                        data_list.append(
                            dict(
                                obs=obs_flatten,
                                action=action,
                                reward=reward,
                                done=done,
                                next_obs=next_obs_flatten,
                            )
                        )
                        obs = next_obs
                self.last_state_obs = self.collect_env.reset().observation
                self.last_state_done = False
            return data_list
        elif num_steps is not None:
            data_list = []
            with torch.no_grad():
                for i in range(num_steps):
                    if self.last_state_done:
                        self.last_state_obs = self.collect_env.reset().observation
                        self.last_state_done = False
                    if random_policy:
                        action = np.random.uniform(
                            self.action_space.minimum,
                            self.action_space.maximum,
                            size=self.action_space.shape,
                        )
                    else:
                        action = policy(obs)
                    time_step = self.collect_env.step(action)
                    next_obs = time_step.observation
                    reward = time_step.reward
                    done = time_step.last()
                    data_list.append(
                        dict(
                            obs=obs,
                            action=action,
                            reward=reward,
                            done=done,
                            next_obs=next_obs,
                        )
                    )
                    self.last_state_obs = next_obs
                    self.last_state_done = done
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

        env = DeepMindControlVisualEnv(name=f"{self.domain_name}_{self.task_name}")

        def render_env(env, render_args):
            # TODO: support different render modes
            render_output = env.render(
                **render_args,
            )
            return render_output

        eval_results = []

        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                step = 0
                time_step = env.reset()
                obs = time_step["image"]
                obs_stack_t = np.stack([obs] * self.stack_frames, axis=0)
                if render:
                    render_output.append(render_env(env, render_args))
                done = False

                while not done:

                    action = policy(obs_stack_t)
                    time_step = env.step(action)
                    next_obs = time_step["image"]

                    reward = time_step["reward"]
                    done = time_step["is_last"] or time_step["is_terminal"]
                    discount = 1.0 - int(time_step["is_terminal"])
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
                            discount=discount,
                        )
                    )
                    obs = next_obs

                    obs_stack_t = np.concatenate(
                        [obs_stack_t[1:], np.expand_dims(next_obs, axis=0)], axis=0
                    )

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
