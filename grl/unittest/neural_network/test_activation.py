# Test grl/neural_network/activation.py


def test_activation():
    import torch
    from torch import nn

    from grl.neural_network.activation import Swish, get_activation

    assert type(get_activation("mish")) == nn.Mish
    assert type(get_activation("tanh")) == nn.Tanh
    assert type(get_activation("relu")) == nn.ReLU
    assert type(get_activation("softplus")) == nn.Softplus
    assert type(get_activation("elu")) == nn.ELU
    assert type(get_activation("silu")) == nn.SiLU
    assert type(get_activation("swish")) == Swish
    assert get_activation("square")(10) == 100
    assert get_activation("identity")(100) == 100

    try:
        get_activation("unknown")
    except ValueError as e:
        assert str(e) == "Unknown activation function unknown"
