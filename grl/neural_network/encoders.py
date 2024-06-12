import math

import numpy as np
import torch
import torch.nn as nn


def get_encoder(type: str):
    """
    Overview:
        Get the encoder module by the encoder type.
    Arguments:
        type (:obj:`str`): The encoder type.
    """

    if type.lower() in ENCODERS:
        return ENCODERS[type.lower()]
    else:
        raise ValueError(f"Unknown encoder type: {type}")


class GaussianFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding time variable.
        This module is used as the encoder of time in generative models such as diffusion model.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

            \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierProjectionEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding variables.
        This module can be seen as a generalization of GaussianFourierProjectionTimeEncoder for encoding multi-dimensional variables.
        It transforms the input tensor :math:`x` to a high-dimensional embedding vector :math:`\phi(x)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(x) = [ \sin(x \cdot w_1), \cos(x \cdot w_1), \sin(x \cdot w_2), \cos(x \cdot w_2), \ldots, \sin(x \cdot w_{\text{embed\_dim} / 2}), \cos(x \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, x_shape, flatten=True, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            x_shape (:obj:`tuple`): The shape of the input tensor.
            flatten (:obj:`bool`): Whether to flatten the output tensor afyer applying the encoder.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )
        self.x_shape = x_shape
        self.flatten = flatten

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, D * embed_dim)` if flatten is True, otherwise :math:`(B, D, embed_dim)`.
                where B is batch size, embed_dim is the dimension of the output embedding vector, D is the shape of the input tensor.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(torch.Size([100, 10]))
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # if x shape is (B1, ..., Bn, **x_shape), then the output shape is (B1, ..., Bn, np.prod(x_shape) * embed_dim)
        if self.flatten:
            x_proj = torch.flatten(x_proj, start_dim=-1 - self.x_shape.__len__())

        return x_proj


class ExponentialFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Expoential Fourier Projection Time Encoder.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

            where :math:`w_i` is a random scalar sampled from a uniform distribution, then transformed by exponential function.
        There is an additional MLP layer to transform the frequency embedding:

        .. math::

            \text{MLP}(\phi(t)) = \text{SiLU}(\text{Linear}(\text{SiLU}(\text{Linear}(\phi(t)))))

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Overview:
            Initialize the timestep embedder.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            frequency_embedding_size (:obj:`int`): The size of the frequency embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    # TODO: simplify this function
    @staticmethod
    def timestep_embedding(t, embed_dim, max_period=10000):
        """
        Overview:
            Create sinusoidal timestep embeddings.
        Arguments:
            t (:obj:`torch.Tensor`): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            embed_dim (:obj:`int`): the dimension of the output.
            max_period (:obj:`int`): controls the minimum frequency of the embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embed_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            t (:obj:`torch.Tensor`): Input time step tensor.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


ENCODERS = {
    "GaussianFourierProjectionTimeEncoder".lower(): GaussianFourierProjectionTimeEncoder,
    "GaussianFourierProjectionEncoder".lower(): GaussianFourierProjectionEncoder,
    "ExponentialFourierProjectionTimeEncoder".lower(): ExponentialFourierProjectionTimeEncoder,
    "SinusoidalPosEmb".lower(): SinusoidalPosEmb,
}
