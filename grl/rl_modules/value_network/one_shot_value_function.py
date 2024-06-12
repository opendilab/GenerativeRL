import copy
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.rl_modules.value_network.value_network import DoubleVNetwork


class OneShotValueFunction(nn.Module):
    """
    Overview:
        Value network for one-shot cases.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of one-shot value network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.v_alpha = config.v_alpha
        self.v = DoubleVNetwork(config.DoubleVNetwork)
        self.v_target = copy.deepcopy(self.v).requires_grad_(False)

    def forward(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of one-shot value network.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return self.v(state, condition)

    def compute_double_v(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Return the output of two value networks.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            v1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first value network.
            v2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second value network.
        """
        return self.v.compute_double_v(state, condition=condition)

    def v_loss(
        self,
        state: Union[torch.Tensor, TensorDict],
        value: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the v loss.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            value (:obj:`Union[torch.Tensor, TensorDict]`): The input value.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            v_loss (:obj:`torch.Tensor`): The v loss.
        """

        # Update value function
        targets = value
        v0, v1 = self.v.compute_double_v(state, condition=condition)
        v_loss = (
            torch.nn.functional.mse_loss(v0, targets)
            + torch.nn.functional.mse_loss(v1, targets)
        ) / 2
        return v_loss
