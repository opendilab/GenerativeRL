import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPResNetBlock(nn.Module):
    """
    Overview:
        MLPResNet block for MLPResNet.
        #TODO: add more details about the block.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
        self,
        hidden_dim: int,
        activations: nn.Module,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
    ):
        """
        Overview:
            Initialize the MLPResNetBlock according to arguments.
        Arguments:
            hidden_dim (:obj:`int`): The dimension of the hidden layer.
            activations (:obj:`nn.Module`): The activation function.
            dropout_rate (:obj:`float`, optional): The dropout rate. Default: None.
            use_layer_norm (:obj:`bool`, optional): Whether to use layer normalization. Default: False.
        """

        super(MLPResNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activations = activations
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate is not None and dropout_rate > 0.0
            else None
        )

    def forward(self, x: torch.Tensor):
        """
        Overview:
            Return the output tensor of the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): Input tensor.
        Returns:
            x (:obj:`torch.Tensor`): Output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size and D is the dimension of the input tensor.
        """
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.activations(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    """
    Overview:
        Residual network build with MLP blocks.
        #TODO: add more details about the network.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: nn.Module = nn.ReLU(),
    ):
        """
        Overview:
            Initialize the MLPResNet.
            #TODO: add more details about the network.
        Arguments:
            num_blocks (:obj:`int`): The number of blocks.
            input_dim (:obj:`int`): The dimension of the input tensor.
            output_dim (:obj:`int`): The dimension of the output tensor.
            dropout_rate (:obj:`float`, optional): The dropout rate. Default: None.
            use_layer_norm (:obj:`bool`, optional): Whether to use layer normalization. Default: False.
            hidden_dim (:obj:`int`, optional): The dimension of the hidden layer. Default: 256.
            activations (:obj:`nn.Module`, optional): The activation function. Default: nn.ReLU().
        """
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim + 128, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    self.hidden_dim,
                    self.activations,
                    self.dropout_rate,
                    self.use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            Return the output tensor of the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): Input tensor.
        Returns:
            x (:obj:`torch.Tensor`): Output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size and D is the dimension of the input tensor.
        """
        x = self.fc(x)

        for block in self.blocks:
            x = block(x)

        x = self.activations(x)
        x = self.out_fc(x)

        return x
