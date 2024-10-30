import collections.abc
import math
from functools import partial
from itertools import repeat
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network import get_module
from grl.neural_network.encoders import ExponentialFourierProjectionTimeEncoder


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


class Mlp(nn.Module):
    """
    Overview:
        MLP as used in Vision Transformer, MLP-Mixer and related networks.
        This module is based on the implementation in "timm.models.vision_transformer.Mlp".
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        """
        Overview:
            Initialize the MLP.
        Arguments:
            in_features (:obj:`int`): The number of input features.
            hidden_features (:obj:`int`, optional): The number of hidden features.
            out_features (:obj:`int`, optional): The number of output features.
            act_layer (:obj:`nn.Module`, optional): The activation layer.
            norm_layer (:obj:`nn.Module`, optional): The normalization layer.
            bias (:obj:`bool`, optional): Whether to use bias.
            drop (:obj:`float`, optional): The dropout probability.
            use_conv (:obj:`bool`, optional): Whether to use convolution.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _ntuple(2)(bias)
        drop_probs = _ntuple(2)(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """
        Overview:
            Forward pass of the MLP.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """
    Overview:
        2D Image to Patch Embedding.
        This module is based on the implementation in "timm.models.vision_transformer.PatchEmbed".
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """
        Overview:
            Initialize the Patch Embedding.
        Arguments:
            img_size (:obj:`Optional[int]`, defaults to 224): The input image size.
            patch_size (:obj:`int`, defaults to 16): The patch size.
            in_chans (:obj:`int`, defaults to 3): The number of input channels.
            embed_dim (:obj:`int`, defaults to 768): The embedding dimension.
            norm_layer (:obj:`Optional[Callable]`, defaults to None): The normalization layer.
            flatten (:obj:`bool`, defaults to True): Whether to flatten the spatial dimensions.
            bias (:obj:`bool`, defaults to True): Whether to use bias.
            strict_img_size (:obj:`bool`, defaults to True): Whether to strictly enforce the image size.
            dynamic_img_pad (:obj:`bool`, defaults to False): Whether to dynamically pad the image.
        """
        super().__init__()
        self.patch_size = _ntuple(2)(patch_size)
        if img_size is not None:
            self.img_size = _ntuple(2)(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.output_fmt = "NCHW"
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Overview:
            Forward pass of the Patch Embedding.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        """
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert (
                    H == self.img_size[0]
                ), f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert (
                    W == self.img_size[1]
                ), f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            elif not self.dynamic_img_pad:
                assert (
                    H % self.patch_size[0] == 0
                ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert (
                    W % self.patch_size[1] == 0
                ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    Overview:
        Multi-head self attention.
        This module is based on the implementation in "timm.models.vision_transformer.Attention".
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """
        Overview:
            Initialize the Attention module.
        Arguments:
            dim (:obj:`int`): The input dimension.
            num_heads (:obj:`int`, defaults to 8): The number of attention heads.
            qkv_bias (:obj:`bool`, defaults to False): Whether to use bias in the qkv projection.
            qk_norm (:obj:`bool`, defaults to False): Whether to use normalization for qk.
            attn_drop (:obj:`float`, defaults to 0.0): The dropout probability for attention.
            proj_drop (:obj:`float`, defaults to 0.0): The dropout probability for projection.
            norm_layer (:obj:`nn.Module`, defaults to nn.LayerNorm): The normalization layer.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the Attention module.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Modulate the input tensor x with the shift and scale tensors.
    Arguments:
        x (:obj:`torch.Tensor`): The input tensor.
        shift (:obj:`torch.Tensor`): The shift tensor.
        scale (:obj:`torch.Tensor`): The scale tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    """
    Overview:
        Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Interfaces:
        ``__init__``, ``token_drop``, ``forward``
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ):
        """
        Overview:
            Initialize the label embedder.
        Arguments:
            num_classes (:obj:`int`): The number of classes.
            hidden_size (:obj:`int`): The hidden size.
            dropout_prob (:obj:`float`, defaults to 0.1): The dropout probability.
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ):
        """
        Overview:
            Drops labels to enable classifier-free guidance.
        Arguments:
            labels (:obj:`torch.Tensor`): The input labels.
            force_drop_ids (:obj:`torch.Tensor`, optional): The force drop ids.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ):
        """
        Overview:
            Embeds the input labels.
        Arguments:
            labels (:obj:`torch.Tensor`): The input labels.
            train (:obj:`bool`, defaults to True): Whether to train the model.
            force_drop_ids (:obj:`torch.Tensor`, optional): The force drop ids.
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_3d_pos_embed(
    embed_dim: int,
    grid_num: List[int],
):
    """
    Overview:
        Get 3D positional embeddings for 3D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension of embeddings for each grid.
        grid_num (:obj:`List[int]`): The number of the grid in each dimension.
    """
    assert len(grid_num) == 3
    grid_num_sum = grid_num[0] + grid_num[1] + grid_num[2]
    assert (
        embed_dim % grid_num_sum == 0
    ), f"Embedding dimension {embed_dim} must be divisible by the total grid size {grid_num_sum}."
    embed_dim_per_grid = embed_dim // grid_num_sum
    grid_0 = np.arange(grid_num[0], dtype=np.float32)
    grid_1 = np.arange(grid_num[1], dtype=np.float32)
    grid_2 = np.arange(grid_num[2], dtype=np.float32)

    grid = np.meshgrid(grid_1, grid_0, grid_2)  # here w goes first
    grid = np.stack(
        [grid[1], grid[0], grid[2]], axis=0
    )  # grid is of shape (3, grid_num[0], grid_num[1], grid_num[2]) or (3, T, H, W)

    # emb_i of shape (embed_dim_per_grid*grid_num[i], total_grid_num = grid_num[0]*grid_num[1]*grid_num[2])
    emb_0 = get_sincos_pos_embed_from_grid(embed_dim_per_grid * grid_num[0], grid[0])
    emb_1 = get_sincos_pos_embed_from_grid(embed_dim_per_grid * grid_num[1], grid[1])
    emb_2 = get_sincos_pos_embed_from_grid(embed_dim_per_grid * grid_num[2], grid[2])

    # emb is of shape (total_grid_num, embed_dim)
    emb = np.concatenate([emb_0, emb_1, emb_2], axis=-1)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
):
    """
    Overview:
        Get 2D positional embeddings for 2D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension for each position.
        grid_size (:obj:`int`): The size of the grid.
        cls_token (:obj:`bool`, defaults to False): Whether to include the class token.
        extra_tokens (:obj:`int`, defaults to 0): The number of extra tokens.
    Returns:
        pos_embed (:obj:`np.ndarray`): The positional embeddings.
    Shapes:
        pos_embed (:obj:`np.ndarray`): [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_pos_embed(
    embed_dim: int,
    grid_num: int,
):
    """
    Overview:
        Get 1D positional embeddings for 1D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension of embeddings for each grid.
        grid_num (:obj:`int`): The number of the grid in each dimension.
    """
    grid = np.arange(grid_num, dtype=np.float32)
    emb = get_sincos_pos_embed_from_grid(embed_dim, grid)
    return emb


def get_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray,
):
    """
    Overview:
        Get positional embeddings for 1D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension for each position.
        pos (:obj:`np.ndarray`): The input positions.
    Returns:
        emb (:obj:`np.ndarray`): The positional embeddings.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("...,d->...d", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray,
):
    """
    Overview:
        Get positional embeddings for 1D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension for each position.
        pos (:obj:`np.ndarray`): The input positions.
    Returns:
        emb (:obj:`np.ndarray`): The positional embeddings.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def meshgrid_3d_pos(grid_num: List[int]):
    """
    Overview:
        Get 3D position for 3D data.
    Arguments:
        grid_num (:obj:`List[int]`): The number of the grid in each dimension.
    """
    assert len(grid_num) == 3
    grid_0 = np.arange(grid_num[0], dtype=np.float32)
    grid_1 = np.arange(grid_num[1], dtype=np.float32)
    grid_2 = np.arange(grid_num[2], dtype=np.float32)

    grid = np.meshgrid(grid_1, grid_0, grid_2)  # here w goes first
    grid = np.stack(
        [grid[1], grid[0], grid[2]], axis=0
    )  # grid is of shape (3, grid_num[0], grid_num[1], grid_num[2]) or (3, T, H, W)

    return grid


class DiTBlock(nn.Module):
    """
    Overview:
        A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs
    ):
        """
        Overview:
            Initialize the DiT block.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            num_heads (:obj:`int`): The number of attention heads.
            mlp_ratio (:obj:`float`, defaults to 4.0): The hidden size of the MLP with respect to the hidden size of Attention.
            block_kwargs (:obj:`dict`): The keyword arguments for the attention block.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.tensor, c: torch.tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    Overview:
        The final layer of DiT.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        """
        Overview:
            Initialize the final layer.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            patch_size (:obj:`int`): The patch size.
            out_channels (:obj:`int`): The number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.tensor, c: torch.tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Overview:
        Diffusion model with a Transformer backbone.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        condition: bool = True,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            input_size (:obj:`int`, defaults to 32): The input size.
            patch_size (:obj:`int`, defaults to 2): The patch size.
            in_channels (:obj:`int`, defaults to 4): The number of input channels.
            hidden_size (:obj:`int`, defaults to 1152): The hidden size.
            depth (:obj:`int`, defaults to 28): The depth.
            num_heads (:obj:`int`, defaults to 16): The number of attention heads.
            mlp_ratio (:obj:`float`, defaults to 4.0): The hidden size of the MLP with respect to the hidden size of Attention.
            class_dropout_prob (:obj:`float`, defaults to 0.1): The class dropout probability.
            num_classes (:obj:`int`, defaults to 1000): The number of classes.
            learn_sigma (:obj:`bool`, defaults to True): Whether to learn sigma.
        """
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition = condition
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        if condition == True:
            self.y_embedder = LabelEmbedder(
                num_classes, hidden_size, class_dropout_prob
            )
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.condition == True:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            imgs (:obj:`torch.Tensor`): The output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): (N, T, patch_size**2 * C)
            imgs (:obj:`torch.Tensor`): (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
    ):
        """
        Overview:
            Forward pass of DiT.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition, self.training)  # (N, D)
            c = t + y  # (N, D)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        cfg_scale: float = 1.0,
    ):
        """
        Overview:
            Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
            cfg_scale (:obj:`float`, defaults to 1.0): The scale for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, condition)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


DiT2D = DiT
DiT_2D = DiT


class FinalLayer3D(nn.Module):
    """
    Overview:
        The final layer of DiT for 3D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Union[int, List[int], Tuple[int]],
        out_channels: Union[int, List[int], Tuple[int]],
    ):
        """
        Overview:
            Initialize the final layer.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            patch_size (:obj:`Union[int, List[int], Tuple[int]]`): The patch size of each token in attention layer.
            out_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of output channels.
        """
        super().__init__()
        assert (
            isinstance(patch_size, (list, tuple))
            and len(patch_size) == 3
            or isinstance(patch_size, int)
        )
        if isinstance(patch_size, int):
            self.patch_size = [patch_size] * 3
        else:
            self.patch_size = list(patch_size)
        assert isinstance(out_channels, (list, tuple)) or isinstance(out_channels, int)
        if isinstance(out_channels, int):
            self.out_channels = [out_channels]
        else:
            self.out_channels = list(out_channels)

        output_dim = np.prod(self.patch_size) * np.prod(self.out_channels)

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Overview:
            Forward pass of the final layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (N, total_patches, hidden_size).
            c (:obj:`torch.Tensor`): The conditioning tensor.
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (N, total_patches, patch_size[0] * patch_size[1] * patch_size[2] * **out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Patchify3D(nn.Module):
    """
    Overview:
        Patchify the input tensor of shape (T, H, W) of attention layer.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        channel_size: Union[int, List[int]] = [3],
        data_size: List[int] = [32, 32, 32],
        patch_size: List[int] = [2, 2, 2],
        hidden_size: int = 768,
        bias: bool = False,
        convolved: bool = False,
    ):
        """
        Overview:
            Initialize the patchify layer.
        Arguments:
            channel_size (:obj:`Union[int, List[int]]`): The number of input channels, defaults to 3.
            data_size (:obj:`List[int]`): The input size of data, defaults to [32, 32, 32].
            patch_size (:obj:`List[int]`): The patch size of each token for attention layer, defaults to [2, 2, 2].
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 768.
            bias (:obj:`bool`): Whether to use bias, defaults to False.
            convolved (:obj:`bool`): Whether to use fully connected layer for all channels, defaults to False.
        """
        super().__init__()
        assert isinstance(data_size, (list, tuple)) or isinstance(data_size, int)
        self.channel_size = (
            list(channel_size)
            if isinstance(channel_size, (list, tuple))
            else [channel_size]
        )
        self.patch_size = patch_size

        in_channels = 1
        for i in self.channel_size:
            in_channels *= i

        self.num_patches = 1
        for i in range(3):
            self.num_patches *= data_size[i] // patch_size[i]

        if convolved:
            self.proj = nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            self.proj = nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                groups=in_channels,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the patchify layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (B, C, T, H, W).
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (B, T' * H'* W', hidden_size). \
            where T' = T // patch_size[0], H' = H // patch_size[1], W' = W // patch_size[2].
        """

        # x: (B, (C1, C2), T, H, W) # x.reshape(shape=(x.shape[0], *self.channel_size, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = x.flatten(start_dim=1, end_dim=-4)
        # x: (B, C1 * C2, T, H, W)
        x = self.proj(x)
        return x


class DiT3D(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for data of 3D shape.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        patch_block_size: Union[List[int], Tuple[int]] = [10, 32, 32],
        patch_size: Union[int, List[int], Tuple[int]] = 2,
        in_channels: Union[int, List[int], Tuple[int]] = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        convolved: bool = False,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            patch_block_size (:obj:`Union[List[int], Tuple[int]]`): The size of patch block, defaults to [10, 32, 32].
            patch_size (:obj:`Union[int, List[int], Tuple[int]]`): The patch size of each token in attention layer, defaults to 2.
            in_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of input channels, defaults to 4.
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 1152.
            depth (:obj:`int`): The depth of transformer, defaults to 28.
            num_heads (:obj:`int`): The number of attention heads, defaults to 16.
            mlp_ratio (:obj:`float`): The hidden size of the MLP with respect to the hidden size of Attention, defaults to 4.0.
            learn_sigma (:obj:`bool`): Whether to learn sigma, defaults to True.
            convolved (:obj:`bool`): Whether to use fully connected layer for all channels, defaults to False.
        """
        super().__init__()

        assert (
            isinstance(patch_block_size, (list, tuple))
            and len(patch_block_size) == 3
            or isinstance(patch_block_size, int)
        )
        self.patch_block_size = (
            list(patch_block_size)
            if isinstance(patch_block_size, (list, tuple))
            else [patch_block_size] * 3
        )
        assert (
            isinstance(patch_size, (list, tuple))
            and len(patch_size) == 3
            or isinstance(patch_size, int)
        )
        self.patch_size = (
            list(patch_size)
            if isinstance(patch_size, (list, tuple))
            else [patch_size] * 3
        )
        for i in range(3):
            assert (
                self.patch_block_size[i] % self.patch_size[i] == 0
            ), f"Patch block size {self.patch_block_size[i]} should be divisible by patch size {self.patch_size[i]}."
        self.patch_grid_num = [
            self.patch_block_size[i] // self.patch_size[i] for i in range(3)
        ]

        self.learn_sigma = learn_sigma
        assert isinstance(in_channels, (list, tuple)) or isinstance(in_channels, int)
        self.in_channels = (
            list(in_channels)
            if isinstance(in_channels, (list, tuple))
            else [in_channels]
        )
        self.out_channels = in_channels * 2 if learn_sigma else self.in_channels

        self.num_heads = num_heads

        self.x_embedder = Patchify3D(
            in_channels,
            patch_block_size,
            patch_size,
            hidden_size,
            bias=True,
            convolved=convolved,
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_3d_pos_embed(
            embed_dim=hidden_size, grid_num=self.patch_grid_num
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer3D(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the output tensor of attention layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (N, total_patches = T' * H' * W', patch_size[0] * patch_size[1] * patch_size[2] * C)
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (N, T, C, H, W).
        """

        x = x.reshape(
            shape=(
                x.shape[0],
                self.patch_grid_num[0],
                self.patch_grid_num[1],
                self.patch_grid_num[2],
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
                np.prod(self.out_channels),
            )
        )
        x = torch.einsum("nthwpqr...->ntp...hqwr", x)
        x = x.reshape(
            shape=(
                x.shape[0],
                self.patch_grid_num[0] * self.patch_size[0],
                *self.out_channels,
                self.patch_grid_num[1] * self.patch_size[1],
                self.patch_grid_num[2] * self.patch_size[2],
            )
        )

        return x

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
    ):
        """
        Overview:
            Forward pass of DiT for 3D data.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of inputs with spatial information (originally at t=0 it is tensor of videos or latent representations of videos).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        # x is of shape (N, T, C, H, W), reshape to (N, C, T, H, W)
        x = torch.einsum("nt...hw->n...thw", x)
        x = self.x_embedder(x) + torch.einsum("tHWh->htHW", self.pos_embed)
        x = x.reshape(shape=(x.shape[0], x.shape[1], -1))
        x = torch.einsum(
            "nhs->nsh", x
        )  # (N, total_patches, hidden_size), where total_patches = T' * H' * W' = T * H * W / patch_size[0] * patch_size[1] * patch_size[2]
        t = self.t_embedder(t)  # (N, hidden_size)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition, self.training)  # (N, hidden_size)
            c = t + y  # (N, hidden_size)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, total_patches, hidden_size)
        x = self.final_layer(
            x, c
        )  # (N, total_patches, patch_size[0] * patch_size[1] * patch_size[2] * C)
        x = self.unpatchify(x)  # (N, T, C, H, W)
        return x


DiT_3D = DiT3D


class FinalLayer1D(nn.Module):
    """
    Overview:
        The final layer of DiT for 1D data.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, hidden_size: int, out_channels: int):
        """
        Overview:
            Initialize the final layer.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            out_channels (:obj:`int`): The number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.tensor, c: torch.tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT1D(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for 1D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        token_size: int,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        condition_embedder: EasyDict = None,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            in_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of input channels, defaults to 4.
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 1152.
            depth (:obj:`int`): The depth of transformer, defaults to 28.
            num_heads (:obj:`int`): The number of attention heads, defaults to 16.
            mlp_ratio (:obj:`float`): The hidden size of the MLP with respect to the hidden size of Attention, defaults to 4.0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.num_heads = num_heads

        self.x_embedder = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            groups=in_channels,
            bias=False,
        )
        if condition_embedder:
            self.y_embedder = get_module(condition_embedder.type)(
                **condition_embedder.args
            )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_1d_pos_embed(embed_dim=hidden_size, grid_num=token_size)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer1D(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
    ):
        """
        Overview:
            Forward pass of DiT for 3D data.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of inputs with spatial information (originally at t=0 it is tensor of videos or latent representations of videos).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        # x is of shape (N, T, C), reshape to (N, C, T)
        x = torch.einsum("ntc->nct", x)
        x = self.x_embedder(x) + torch.einsum("th->ht", self.pos_embed)

        t = self.t_embedder(t)  # (N, hidden_size)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition, self.training)  # (N, hidden_size)
            c = t + y  # (N, hidden_size)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, total_patches, hidden_size)
        x = self.final_layer(x, c)  # (N, total_patches, C)
        return x


DiT_1D = DiT1D
