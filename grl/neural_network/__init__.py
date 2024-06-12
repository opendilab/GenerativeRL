from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network.activation import get_activation
from grl.neural_network.encoders import get_encoder
from grl.neural_network.residual_network import MLPResNet


def register_module(module: nn.Module, name: str):
    """
    Overview:
        Register the module to the module dictionary.
    Arguments:
        - module (:obj:`nn.Module`): The module to be registered.
        - name (:obj:`str`): The name of the module.
    """
    global MODULES
    if name.lower() in MODULES:
        raise KeyError(f"Module {name} is already registered.")
    MODULES[name.lower()] = module


def get_module(type: str):
    if type.lower() in MODULES:
        return MODULES[type.lower()]
    else:
        raise ValueError(f"Unknown module type: {type}")


def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:
    """
    Overview:
        Construct the corresponding normalization module. For beginners,
        refer to [this article](https://zhuanlan.zhihu.com/p/34879333) to learn more about batch normalization.
    Arguments:
        - norm_type (:obj:`str`): Type of the normalization. Currently supports ['BN', 'LN', 'IN', 'SyncBN'].
        - dim (:obj:`Optional[int]`): Dimension of the normalization, applicable when norm_type is in ['BN', 'IN'].
    Returns:
        - norm_func (:obj:`nn.Module`): The corresponding batch normalization function.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ["BN", "IN"]:
            key = norm_type + str(dim)
        elif norm_type in ["LN", "SyncBN"]:
            key = norm_type
        else:
            raise NotImplementedError(
                "not support indicated dim when creates {}".format(norm_type)
            )
    norm_func = {
        "BN1": nn.BatchNorm1d,
        "BN2": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
        "IN1": nn.InstanceNorm1d,
        "IN2": nn.InstanceNorm2d,
        "SyncBN": nn.SyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))


def sequential_pack(layers: List[nn.Module]) -> nn.Sequential:
    """
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`List[nn.Module]`): The input list of layers.
    Returns:
        - seq (:obj:`nn.Sequential`): Packed sequential container.
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False,
):
    """
    Overview:
        Create a multi-layer perceptron using fully-connected blocks with activation, normalization, and dropout,
        optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - layer_num (:obj:`int`): Number of layers.
        - layer_fn (:obj:`Callable`, optional): Layer function.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): The type of the normalization.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block. Default is False.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
        - output_activation (:obj:`bool`, optional): Whether to use activation in the output layer. If True, \
            we use the same activation as front layers. Default is True.
        - output_norm (:obj:`bool`, optional): Whether to use normalization in the output layer. If True, \
            we use the same normalization as front layers. Default is True.
        - last_linear_layer_init_zero (:obj:`bool`, optional): Whether to use zero initializations for the last \
            linear layer (including w and b), which can provide stable zero outputs in the beginning, \
            usually used in the policy network in RL settings.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the multi-layer perceptron.

    .. note::
        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    """
    In the final layer of a neural network, whether to use normalization and activation are typically determined
    based on user specifications. These specifications depend on the problem at hand and the desired properties of
    the model's output.
    """
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)


class TemporalSpatialResBlock(nn.Module):
    """
    Overview:
        Residual Block using MLP layers for both temporal and spatial input.
        t → time_mlp  →  h1 → dense2 → h2 → out
                       ↗+                ↗+
        x →  dense1 →                  ↗
          ↘                          ↗
            → modify_x →   →   →   →
    """

    def __init__(self, input_dim, output_dim, t_dim=128, activation=torch.nn.SiLU()):
        """
        Overview:
            Init the temporal spatial residual block.
        Arguments:
            - input_dim (:obj:`int`): The number of channels in the input tensor.
            - output_dim (:obj:`int`): The number of channels in the output tensor.
            - t_dim (:obj:`int`): The dimension of the temporal input.
            - activation (:obj:`nn.Module`): The optional activation function.
        """
        super().__init__()
        # temporal input is the embedding of time, which is a Gaussian Fourier Feature tensor
        self.time_mlp = nn.Sequential(
            activation,
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim), activation)
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim), activation)
        self.modify_x = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, t, x) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - t (:obj:`torch.Tensor`): The temporal input tensor.
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)


class TemporalSpatialResidualNet(nn.Module):
    """
    Overview:
        Temporal Spatial Residual Network using multiple TemporalSpatialResBlock.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        output_dim: int,
        t_dim: int,
        input_dim: int = None,
        condition_dim: int = None,
        condition_hidden_dim: int = None,
        t_condition_hidden_dim: int = None,
    ):
        """
        Overview:
            Initiate the temporal spatial residual network.
        Arguments:
            - hidden_sizes (:obj:`List[int]`): The list of hidden sizes.
            - output_dim (:obj:`int`): The number of channels in the output tensor.
            - t_dim (:obj:`int`): The dimension of the temporal input.
            - condition_dim (:obj:`int`, optional): The number of channels in the condition tensor. Default is None.
            - condition_hidden_dim (:obj:`int`, optional): The number of channels in the hidden condition tensor. \
                Default is None.
            - t_condition_hidden_dim (:obj:`int`, optional): The number of channels in the hidden temporal condition tensor. \
                Default is None.
        """
        super().__init__()
        if input_dim is None:
            input_dim = output_dim
        if condition_dim is None or condition_dim <= 0:
            condition_hidden_dim = 0
            t_condition_dim = t_dim
            t_condition_hidden_dim = t_dim
        else:
            t_condition_dim = t_dim + condition_hidden_dim
            t_condition_hidden_dim = (
                t_condition_hidden_dim
                if t_condition_hidden_dim is not None
                else t_condition_dim
            )
            self.pre_sort_condition = nn.Sequential(
                nn.Linear(condition_dim, condition_hidden_dim), torch.nn.SiLU()
            )
        self.sort_t = nn.Sequential(
            nn.Linear(t_condition_dim, t_condition_hidden_dim),
            torch.nn.SiLU(),
            nn.Linear(t_condition_hidden_dim, t_condition_hidden_dim),
        )
        self.first_block = TemporalSpatialResBlock(
            input_dim, hidden_sizes[0], t_dim=t_condition_hidden_dim
        )
        self.down_block = nn.ModuleList(
            [
                TemporalSpatialResBlock(
                    hidden_sizes[i], hidden_sizes[i + 1], t_dim=t_condition_hidden_dim
                )
                for i in range(len(hidden_sizes) - 1)
            ]
        )
        self.middle_block = TemporalSpatialResBlock(
            hidden_sizes[-1], hidden_sizes[-1], t_dim=t_condition_hidden_dim
        )
        self.up_block = nn.ModuleList(
            [
                TemporalSpatialResBlock(
                    hidden_sizes[i], hidden_sizes[i], t_dim=t_condition_hidden_dim
                )
                for i in range(len(hidden_sizes) - 2, -1, -1)
            ]
        )
        self.last_block = nn.Linear(hidden_sizes[0] * 2, output_dim)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of the temporal spatial residual network.
        Arguments:
            - t (:obj:`torch.Tensor`): The temporal input tensor.
            - x (:obj:`torch.Tensor`): The input tensor.
            - condition (:obj:`torch.Tensor`, optional): The condition tensor. Default is None.
        """

        if condition is not None:
            t_condition = torch.cat([t, self.pre_sort_condition(condition)], dim=-1)
        else:
            t_condition = t
        t_condition_embedding = self.sort_t(t_condition)
        d0 = self.first_block(t_condition_embedding, x)
        d = [d0]
        for i, block in enumerate(self.down_block):
            d_i = block(t_condition_embedding, d[i])
            d.append(d_i)
        u = self.middle_block(t_condition_embedding, d[-1])
        for i, block in enumerate(self.up_block):
            u = block(t_condition_embedding, torch.cat([u, d[-i - 1]], dim=-1))
        return self.last_block(torch.cat([u, d[0]], dim=-1))


class ConcatenateLayer(nn.Module):
    """
    Overview:
        Concatenate the input tensors along the last dimension.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(self):
        """
        Overview:
            Initiate the concatenate layer.
        """
        super().__init__()

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the concatenated tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        return torch.cat(x, dim=-1)


class MultiLayerPerceptron(nn.Module):
    """
    Overview:
        Multi-layer perceptron using fully-connected layers with activation, dropout, and layernorm.
        x -> fc1 -> act1 -> dropout -> layernorm -> ... -> fcn -> actn -> out
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        activation: Union[str, List[str]],
        dropout: float = None,
        layernorm: bool = False,
        final_activation: str = None,
        scale: float = None,
        shrink: float = None,
    ):
        """
        Overview:
            Initiate the multi-layer perceptron.
        Arguments:
            - hidden_sizes (:obj:`List[int]`): The list of hidden sizes.
            - output_size (:obj:`int`): The number of channels in the output tensor.
            - activation (:obj:`Union[str, List[str]]`): The optional activation function.
            - dropout (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. Default is None.
            - layernorm (:obj:`bool`, optional): Whether to use layernorm in the fully-connected block. Default is False.
            - final_activation (:obj:`str`, optional): The optional activation function in the final layer. Default is None.
            - scale (:obj:`float`, optional): The scale of the output tensor. Default is None.
            - shrink (:obj:`float`, optional): The shrinkage factor of the output tensor. Default is None.
        """
        super().__init__()

        self.model = nn.Sequential()

        for i in range(len(hidden_sizes) - 1):
            self.model.add_module(
                "linear" + str(i), nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )

            if isinstance(activation, list):
                self.model.add_module(
                    "activation" + str(i), get_activation(activation[i])
                )
            else:
                self.model.add_module("activation" + str(i), get_activation(activation))
            if dropout is not None and dropout > 0:
                self.model.add_module("dropout", nn.Dropout(dropout))
            if layernorm:
                self.model.add_module("layernorm", nn.LayerNorm(hidden_sizes[i + 1]))

        self.model.add_module(
            "linear" + str(len(hidden_sizes) - 1),
            nn.Linear(hidden_sizes[-1], output_size),
        )

        if final_activation is not None:
            self.model.add_module("final_activation", get_activation(final_activation))

        if scale is not None:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)
        else:
            self.scale = 1.0

        # shrink the weight of linear layer 'linear'+str(len(hidden_sizes) to it's origin 0.01
        if shrink is not None:
            if final_activation is not None:
                self.model[-2].weight.data.normal_(0, shrink)
            else:
                self.model[-1].weight.data.normal_(0, shrink)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the output of the multi-layer perceptron.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        return self.scale * self.model(x)


class ConcatenateMLP(nn.Module):
    """
    Overview:
        Concatenate the input tensors along the last dimension and then pass through a multi-layer perceptron.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(self, **kwargs):
        """
        Overview:
            Initiate the concatenate MLP.
        Arguments:
            - **kwargs: The keyword arguments for the multi-layer perceptron.
        """
        super().__init__()
        self.model = MultiLayerPerceptron(**kwargs)

    def forward(self, *x):
        """
        Overview:
            Return the output of the concatenate MLP.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        return self.model(torch.cat(x, dim=-1))


class ALLCONCATMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.main = MLPResNet(**kwargs)
        self.t_cond = MultiLayerPerceptron(
            hidden_sizes=[64, 128],
            output_size=128,
            activation="mish",
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:

        embed = self.t_cond(t)
        result = self.main(torch.cat([x, condition, embed], dim=-1))
        return result


from .transformers.dit import DiT, DiT1D, DiT3D


class Sequential(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.ModuleList()

        for key, config in kwargs.items():
            module = get_module(config.type)(**config.args)
            self.model.append(module)

    def forward(self, *x):
        for module in self.model:
            x = module(*x)
        return x


class TimeExtension(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        if len(t.shape) == 0 or t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        return t, x, condition


class IntrinsicModel(nn.Module):
    """
    Overview:
        Intrinsic model of generative model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        # TODO

        super().__init__()

        self.config = config
        assert hasattr(config, "backbone"), "backbone must be specified in config"

        self.model = torch.nn.ModuleDict()
        if hasattr(config, "t_encoder"):
            self.model["t_encoder"] = get_encoder(config.t_encoder.type)(
                **config.t_encoder.args
            )
        else:
            self.model["t_encoder"] = torch.nn.Identity()
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
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of the model at time t given the initial state.
        """

        if condition is not None:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            condition = self.model["condition_encoder"](condition)
            output = self.model["backbone"](t, x, condition)
        else:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            output = self.model["backbone"](t, x)

        return output


MODULES = {
    "Sequential".lower(): Sequential,
    "TimeExtension".lower(): TimeExtension,
    "IntrinsicModel".lower(): IntrinsicModel,
    "ConcatenateLayer".lower(): ConcatenateLayer,
    "MultiLayerPerceptron".lower(): MultiLayerPerceptron,
    "ConcatenateMLP".lower(): ConcatenateMLP,
    "ALLCONCATMLP".lower(): ALLCONCATMLP,
    "TemporalSpatialResidualNet".lower(): TemporalSpatialResidualNet,
    "DiT".lower(): DiT,
    "DiT_3D".lower(): DiT3D,
    "DiT_2D".lower(): DiT,
    "DiT_1D".lower(): DiT1D,
    "DiT3D".lower(): DiT3D,
    "DiT2D".lower(): DiT,
    "DiT1D".lower(): DiT1D,
}
