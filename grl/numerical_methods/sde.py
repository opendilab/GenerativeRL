from typing import Callable, Union

import torch
from easydict import EasyDict
from tensordict import TensorDict
from torch import nn


class SDE:
    """
    Overview:
        Base class for stochastic differential equations.
        The SDE is defined as:

        .. math::
            dx = f(x, t)dt + g(x, t)dW

        where f(x, t) is the drift term, g(x, t) is the diffusion term, and dW is the Wiener process.

    Interfaces:
        ``__init__``
    """

    def __init__(
        self,
        drift: Union[nn.Module, Callable] = None,
        diffusion: Union[nn.Module, Callable] = None,
    ):
        self.drift = drift
        self.diffusion = diffusion
