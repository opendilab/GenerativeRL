from typing import Dict, Union

import numpy as np
import torch
from easydict import EasyDict

from grl.agents import obs_transform, action_transform


class GPAgent:
    """
    Overview:
        The agent trained for generative policies.
        This class is designed to be used with the ``GMPGAlgorithm`` and ``GMPOAlgorithm``.
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

        obs = obs_transform(obs, self.device)

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
                    solver_config=(
                        self.config.solver_config
                        if hasattr(self.config, "solver_config")
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

        action = action_transform(action, return_as_torch_tensor)

        return action
