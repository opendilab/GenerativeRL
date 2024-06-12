from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict


class NoiseFunction:
    """
    Overview:
        Model of noise function in diffusion model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        model_type: str,
        process: object,
    ):
        """
        Overview:
            Initialize the noise function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`object`): The process.
        """

        self.model_type = model_type
        self.process = process
        # TODO: add more types
        assert self.model_type in [
            "data_prediction_function",
            "noise_function",
            "score_function",
            "velocity_function",
            "denoiser_function",
        ], "Unknown type of ScoreFunction {}".format(type)

    def forward(
        self,
        model: Union[Callable, nn.Module],
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return noise function of the model at time t given the initial state.
            .. math::
                \frac{- \sigma(t) x_t + \sigma^2(t) \nabla_{x_t} \log p_{\theta}(x_t)}{s(t)}

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if self.model_type == "noise_function":
            return model(t, x, condition)
        elif self.model_type == "score_function":
            return -model(t, x, condition) * self.process.std(t, x)
        elif self.model_type == "velocity_function":
            return (
                (model(t, x, condition) - self.process.drift(t, x))
                * 2.0
                * self.process.std(t, x)
                / self.process.diffusion_squared(t, x)
            )
        elif self.model_type == "data_prediction_function":
            return (
                x - self.process.scale(t, x) * model(t, x, condition)
            ) / self.process.std(t, x)
        else:
            raise NotImplementedError("Unknown type of noise function {}".format(type))
