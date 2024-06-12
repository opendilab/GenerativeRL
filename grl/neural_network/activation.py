import torch
from torch import nn


class Swish(nn.Module):
    """
    Overview:
        Swish activation function.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    """
    Overview:
        Lambda activation function.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


ACTIVATIONS = {
    "mish": nn.Mish(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


def get_activation(name: str):
    if name not in ACTIVATIONS:
        raise ValueError("Unknown activation function {}".format(name))
    return ACTIVATIONS[name]
