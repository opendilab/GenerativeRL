from typing import Tuple, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network import get_module
from grl.neural_network.encoders import get_encoder


class VNetwork(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict()
        if hasattr(config, "state_encoder"):
            self.model["state_encoder"] = get_encoder(config.state_encoder.type)(
                **config.state_encoder.args
            )
        else:
            self.model["state_encoder"] = torch.nn.Identity()
        if hasattr(config, "condition_encoder"):
            self.model["condition_encoder"] = get_encoder(
                config.condition_encoder.type
            )(**config.condition_encoder.args)
        else:
            self.model["condition_encoder"] = torch.nn.Identity()
        # TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(
            **config.backbone.args
        )

    def forward(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return output of value networks.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            value (:obj:`Union[torch.Tensor, TensorDict]`): The output of value network.
        """

        state_embedding = self.model["state_encoder"](state)
        if condition is not None:
            condition_encoder_embedding = self.model["condition_encoder"](condition)
            return self.model["backbone"](state_embedding, condition_encoder_embedding)
        else:
            return self.model["backbone"](state_embedding)


class DoubleVNetwork(nn.Module):
    """
    Overview:
        Double value network, which has two value networks.
    Interfaces:
        ``__init__``, ``forward``, ``compute_double_v``, ``compute_mininum_v``
    """

    def __init__(self, config: EasyDict):
        super().__init__()

        self.model = torch.nn.ModuleDict()
        self.model["v1"] = VNetwork(config)
        self.model["v2"] = VNetwork(config)

    def compute_double_v(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict],
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

        return self.model["v1"](state, condition), self.model["v2"](state, condition)

    def compute_mininum_v(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two value networks.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            minimum_v (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of value network.
        """

        return torch.min(*self.compute_double_v(state, condition=condition))

    def forward(
        self,
        state: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return the minimum output of two value networks.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            minimum_v (:obj:`Union[torch.Tensor, TensorDict]`): The minimum output of value network.
        """

        return self.compute_mininum_v(state, condition=condition)
