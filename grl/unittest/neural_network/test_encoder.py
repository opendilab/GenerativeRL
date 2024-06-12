# Test grl/neural_network/encoder.py


def test_encoder():
    import torch
    from torch import nn

    from grl.neural_network.encoders import (
        ExponentialFourierProjectionTimeEncoder,
        GaussianFourierProjectionEncoder,
        GaussianFourierProjectionTimeEncoder,
    )

    encoder = GaussianFourierProjectionTimeEncoder(128)
    x = torch.randn(100)
    output = encoder(x)
    assert output.shape == (100, 128)

    encoder = GaussianFourierProjectionEncoder(128, x_shape=(10,), flatten=False)
    x = torch.randn(100, 10)
    output = encoder(x)
    assert output.shape == (100, 10, 128)

    encoder = GaussianFourierProjectionEncoder(128, x_shape=(10,), flatten=True)
    x = torch.randn(100, 10)
    output = encoder(x)
    assert output.shape == (100, 1280)

    encoder = GaussianFourierProjectionEncoder(128, x_shape=(10, 20), flatten=False)
    x = torch.randn(100, 10, 20)
    output = encoder(x)
    assert output.shape == (100, 10, 20, 128)

    encoder = GaussianFourierProjectionEncoder(128, x_shape=(10, 20), flatten=True)
    x = torch.randn(100, 10, 20)
    output = encoder(x)
    assert output.shape == (100, 25600)

    encoder = ExponentialFourierProjectionTimeEncoder(128)
    x = torch.randn(100)
    output = encoder(x)
    assert output.shape == (100, 128)
