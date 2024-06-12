from typing import Tuple, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network import get_module
from grl.neural_network.encoders import get_encoder


class QNetwork(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict()
        if hasattr(config, "action_encoder"):
            self.model["action_encoder"] = get_encoder(config.action_encoder.type)(
                **config.action_encoder.args
            )
        else:
            self.model["action_encoder"] = torch.nn.Identity()
        if hasattr(config, "state_encoder"):
            self.model["state_encoder"] = get_encoder(config.state_encoder.type)(
                **config.state_encoder.args
            )
        else:
            self.model["state_encoder"] = torch.nn.Identity()
        # TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(
            **config.backbone.args
        )

    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return output of Q networks.
        Arguments:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            q (:obj:`Union[torch.Tensor, TensorDict]`): The output of Q network.
        """
        action_embedding = self.model["action_encoder"](action)
        state_embedding = self.model["state_encoder"](state)
        return self.model["backbone"](action_embedding, state_embedding)


class DoubleQNetwork(nn.Module):
    """
    Overview:
        Double Q network, which has two Q networks.
    Interfaces:
        ``__init__``, ``forward``, ``compute_double_q``, ``compute_mininum_q``
    """

    def __init__(self, config: EasyDict):
        super().__init__()

        self.model = torch.nn.ModuleDict()
        self.model["q1"] = QNetwork(config)
        self.model["q2"] = QNetwork(config)

    def compute_double_q(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
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

        return self.model["q1"](action, state), self.model["q2"](action, state)

    def compute_mininum_q(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two Q networks.
        Arguments:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            minimum_q (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of Q network.
        """

        return torch.min(*self.compute_double_q(action, state))

    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two Q networks.
        Arguments:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            minimum_q (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of Q network.
        """

        return self.compute_mininum_q(action, state)
