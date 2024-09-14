from typing import Callable, Dict, List, Union

import numpy as np
import torch 


def partial_observation_rodent(obs_dict):
    # Define the keys you want to keep
    keys_to_keep = [
        'walker/joints_pos',
        'walker/joints_vel',
        'walker/tendons_pos',
        'walker/tendons_vel',
        'walker/appendages_pos',
        'walker/world_zaxis',
        'walker/sensors_accelerometer',
        'walker/sensors_velocimeter',
        'walker/sensors_gyro',
        'walker/sensors_touch',
        'walker/egocentric_camera'
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

    def __init__(
            self,
            domain_name: str,
            task_name: str,
            dict_return=True
        ) -> None:
        """
        Overview:
            Initialize the DeepMindControlEnvSimulator according to the given configuration.
        Arguments:
            domain_name (:obj:`str`): The domain name of the environment.
            task_name (:obj:`str`): The task name of the environment.
            dict_return (:obj:`bool`): Whether to return the observation as a dictionary.
        """
        if domain_name  ==  "rodent" and task_name == "gaps":
            import os
            os.environ['MUJOCO_EGL_DEVICE_ID'] = '0' #we make it for 8 gpus
            from dm_control import composer
            from dm_control.locomotion.examples import basic_rodent_2020
            self.domain_name = domain_name
            self.task_name=task_name
            self.collect_env=basic_rodent_2020.rodent_run_gaps()
            self.action_space = self.collect_env.action_spec()
            self.partial_observation=True
            self.partial_observation_fn=partial_observation_rodent
        else:
            from dm_control import suite
            self.domain_name = domain_name
            self.task_name=task_name
            self.collect_env = suite.load(domain_name, task_name)
            self.action_space = self.collect_env.action_spec()
            self.partial_observation=False

        self.last_state_obs = self.collect_env.reset().observation
        self.last_state_done = False
        self.dict_return=dict_return
        
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
                    while not done :
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
                                action = np.random.uniform(self.action_space.minimum,
                                    self.action_space.maximum,
                                    size=self.action_space.shape)
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
                        action = np.random.uniform(self.action_space.minimum,
                            self.action_space.maximum,
                            size=self.action_space.shape)
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
        if self.domain_name  ==  "rodent" and self.task_name == "gaps":
            import os
            os.environ['MUJOCO_EGL_DEVICE_ID'] = '0' 
            from dm_control import composer
            from dm_control.locomotion.examples import basic_rodent_2020
            env=basic_rodent_2020.rodent_run_gaps()
        else:
            env = suite.load(self.domain_name, self.task_name)
        for i in range(num_episodes):
            if render:
                render_output = []
            data_list = []
            with torch.no_grad():
                step = 0
                time_step = env.reset()
                obs=time_step.observation
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
                                if value.ndim == 2 :
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
