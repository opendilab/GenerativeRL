from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network import get_module
from grl.neural_network.encoders import get_encoder


class IntrinsicModel(nn.Module):
    """
    Overview:
        Intrinsic model of VAE model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the intrinsic model.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()

        self.config = config
        assert hasattr(config, "backbone"), "backbone must be specified in config"

        self.model = torch.nn.ModuleDict()
        if hasattr(config, "x_encoder"):
            self.model["x_encoder"] = get_encoder(config.x_encoder.type)(
                **config.x_encoder.args
            )
        else:
            self.model["x_encoder"] = torch.nn.Identity()
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
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of the model at time t given the initial state.
        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        if condition is not None:
            x = self.model["x_encoder"](x)
            condition = self.model["condition_encoder"](condition)
            output = self.model["backbone"](x, condition)
        else:
            x = self.model["x_encoder"](x)
            output = self.model["backbone"](x)

        return output


class VariationalAutoencoder(nn.Module):
    """
    Overview:
        Variational Autoencoder model.
        This is an in-development model, which is not used in the current version of the codebase.
    Interfaces:
        ``__init__``, ``encode``, ``reparameterize``, ``decode``, ``forward``
    """

    def __init__(self, config: EasyDict):
        super().__init__()

        self.device = config.device
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        # Encoder
        self.encoder = IntrinsicModel(config.encoder)

        # Decoder
        self.decoder = IntrinsicModel(config.decoder)

    def encode(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ):
        mu, logvar = self.encoder(x, condition)
        return mu, logvar

    def reparameterize(
        self,
        mu: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        logvar: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
    ):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(
        self,
        z: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ):
        x = self.decoder(z, condition)
        return x

    def forward(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        return x_recon, mu, logvar
