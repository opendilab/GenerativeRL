from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from easydict import EasyDict


class BaseSimulator:
    """
    Overview:
        A base class for environment simulator in GenerativeRL.
        This class is used to define the interface of environment simulator in GenerativeRL.
    Interfaces:
        ``__init__``, ``collect_episodes``, ``collect_episodes``, ``evaluate``
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize the environment simulator.
        """
        pass

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

        pass

    def collect_steps(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
        num_steps: int = None,
    ) -> List[Dict]:
        """
        Overview:
            Collect several steps using the given policy. The environment will not be reset until the end of the episode.
            Last observation will be stored in this method. The collected information of steps will be returned as a list of dictionaries.
        Arguments:
            policy (:obj:`Union[Callable, torch.nn.Module]`): The policy to collect steps.
            num_episodes (:obj:`int`): The number of episodes to collect.
            num_steps (:obj:`int`): The number of steps to collect.
        """
        pass

    def evaluate(
        self,
        policy: Union[Callable, torch.nn.Module],
        num_episodes: int = None,
    ) -> List[Dict]:
        """
        Overview:
            Evaluate the given policy using the environment. The environment will be reset at the beginning of each episode.
            No history will be stored in this method. The evaluation resultswill be returned as a list of dictionaries.
        """
        pass


class BaseEnv:
    """
    Overview:
        A base class for environment in GenerativeRL.
        This class is used to define the interface of environment in GenerativeRL.
    Interfaces:
        ``__init__``, ``reset``, ``step``, ``render``, ``close``
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize the environment.
        """
        pass

    def reset(self) -> Any:
        """
        Overview:
            Reset the environment and return the initial observation.
        """
        pass

    def step(self, action: Any) -> Any:
        """
        Overview:
            Take an action in the environment and return the next observation, reward, done, and information.
        """
        pass

    def render(self) -> None:
        """
        Overview:
            Render the environment.
        """
        pass

    def close(self) -> None:
        """
        Overview:
            Close the environment.
        """
        pass
