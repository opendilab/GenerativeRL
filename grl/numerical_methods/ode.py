from typing import Callable, Union

import torch
from tensordict import TensorDict
from torch import nn


class ODE:
    """
    Overview:
        Base class for ordinary differential equations.
        The ODE is defined as:

        .. math::
            dx = f(x, t)dt

        where f(x, t) is the drift term.

    Interfaces:
        ``__init__``
    """

    def __init__(
        self,
        drift: Union[nn.Module, Callable] = None,
    ):
        self.drift = drift
