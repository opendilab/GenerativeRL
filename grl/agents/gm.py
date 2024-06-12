from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict


class GPAgent:
    """
    Overview:
        The agent trained for generative policies.
    Interface:
        ``__init__``, ``action``
    """

    def __init__(
        self,
        config: EasyDict,
        model: Union[torch.nn.Module, torch.nn.ModuleDict],
    ):
        """
        Overview:
            Initialize the agent with the configuration and the model.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        """

        self.config = config
        self.device = config.device
        self.model = model.to(self.device)

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor, Dict],
        return_as_torch_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor, Dict]:
        """
        Overview:
            Given an observation, return an action as a numpy array or a torch tensor.
        Arguments:
            obs (:obj:`Union[np.ndarray, torch.Tensor, Dict]`): The observation.
            return_as_torch_tensor (:obj:`bool`): Whether to return the action as a torch tensor.
        Returns:
            action (:obj:`Union[np.ndarray, torch.Tensor, Dict]`): The action.
        """

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, Dict):
            obs = {
                k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()
            }
        elif isinstance(obs, torch.Tensor):
            obs = obs.float().to(self.device)
        else:
            raise ValueError("observation must be a dict, torch.Tensor, or np.ndarray")

        with torch.no_grad():

            # ---------------------------------------
            # Customized inference code ↓
            # ---------------------------------------

            obs = obs.unsqueeze(0)
            action = (
                self.model.sample(
                    condition=obs,
                    t_span=(
                        torch.linspace(0.0, 1.0, self.config.t_span).to(obs.device)
                        if self.config.t_span is not None
                        else None
                    ),
                )
                .squeeze(0)
                .cpu()
                .detach()
                .numpy()
            )

            # ---------------------------------------
            # Customized inference code ↑
            # ---------------------------------------

        if isinstance(action, Dict):
            if return_as_torch_tensor:
                action = {k: v.cpu() for k, v in action.items()}
            else:
                action = {k: v.cpu().numpy() for k, v in action.items()}
        elif isinstance(action, torch.Tensor):
            if return_as_torch_tensor:
                action = action.cpu()
            else:
                action = action.numpy()
        elif isinstance(action, np.ndarray):
            pass
        else:
            raise ValueError("action must be a dict, torch.Tensor, or np.ndarray")

        return action
