from typing import List, Tuple, Union

import torch
import torch.nn as nn

from easydict import EasyDict
from tensordict import TensorDict

from grl.neural_network import get_module
from grl.neural_network.encoders import get_encoder
from grl.generative_models.intrinsic_model import IntrinsicModel
import treetensor


class Scheduler:
    """
    Overview:
        The scheduler of the discrete flow matching model.
    Interfaces:
        ``__init__``, ``k``, ``pt_z_condition_x0_x1``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the scheduler.
        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()
        self.config = config
        self.dimension = config.dimension
        self.unconditional_coupling = (
            True
            if hasattr(config, "unconditional_coupling")
            and config.unconditional_coupling
            else False
        )

        ## self.p_x0 is of shape (dimension, )
        if self.unconditional_coupling:
            self.p_x0 = torch.zeros([self.config.dimension])
            self.p_x0[-1] = 1

        else:
            raise NotImplementedError("Conditional coupling is not implemented yet.")

    def k(self, t):
        """
        Overview:
            The function k(t) in the paper, which is the interpolation function between x0 and x1.
        Arguments:
            t (:obj:`torch.Tensor`): The time.
        """
        return t

    def pt_z_condition_x0_x1(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """
        Overview:
            The probability of the discrete variable z at time t conditioned on x0 and x1.
        Arguments:
            t (:obj:`torch.Tensor`): The time.
            x0 (:obj:`torch.Tensor`): The initial state.
            x1 (:obj:`torch.Tensor`): The final state.
        Returns:
            pt_z_condition_x0_x1 (:obj:`torch.Tensor`): The probability mass of the discrete variable z at time t conditioned on x0 and x1.

            .. math::
                pt(z|x_0, x_1) = (1 - k(t)) * \delta_{x_0}(z) + k(t) * \delta_{x_1}(z)

        Shapes:
            t (:obj:`torch.Tensor`): :math:`(B,)`
            x0 (:obj:`torch.Tensor`): :math:`(B, N)`
            x1 (:obj:`torch.Tensor`): :math:`(B, N)`
            p_t_z_condition_x0_x1 (:obj:`torch.Tensor`): :math:`(B, N, D)`
        """

        # Delta function for x_0
        delta_x0 = self.p_x0.to(x1.device).repeat(x1.shape[0], x1.shape[1], 1)
        # Shape: (B, N, D)

        # Delta function for x_1, change x_1 into onehot encoding
        x1_one_hot = torch.nn.functional.one_hot(
            x1.long(), num_classes=self.dimension
        ).float()  # Shape: (B, N, D)

        return torch.einsum("b,bij->bij", 1 - self.k(t), delta_x0) + torch.einsum(
            "b,bij->bij", self.k(t), x1_one_hot
        )


class DiscreteFlowMatchingModel(nn.Module):
    """
    Overview:
        The discrete flow matching model. Naive implementation of paper "Discrete Flow Matching" <https://arxiv.org/abs/2407.15595>.
    Interfaces:
        ``__init__``, ``forward``, ``sample``, ``flow_matching_loss``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the discrete flow matching model.
        Arguments:
            config (:obj:`EasyDict`): The configuration, which should contain the following keys:
                - model (:obj:`EasyDict`): The configuration of the intrinsic model.
                - scheduler (:obj:`EasyDict`): The configuration of the scheduler.
                - device (:obj:`torch.device`): The device.
                - variable_num (:obj:`int`): The number of variables.
                - dimension (:obj:`int`): The dimension of the discrete variable
        """
        super().__init__()
        self.config = config
        self.device = config.device
        self.variable_num = config.variable_num
        self.dimension = config.dimension

        self.model = IntrinsicModel(config.model.args)
        self.scheduler = Scheduler(config.scheduler)

        self.t_max = 1.0

    def forward(self, x, condition):
        """
        Overview:
            The forward function of the discrete flow matching model.
        Arguments:
            x (:obj:`torch.Tensor`): The state.
            condition (:obj:`torch.Tensor`): The condition.
        """
        pass

    def sample(
        self,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the discrete flow matching model.
        Arguments:
            t_span (:obj:`torch.Tensor`, optional): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`, optional): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`, optional): The initial state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`, optional): The condition.
            with_grad (:obj:`bool`, optional): Whether to keep the gradient.
            solver_config (:obj:`EasyDict`, optional): The configuration of the solver.
        """
        return self.sample_forward_process(
            t_span, batch_size, x_0, condition, with_grad, solver_config
        )[-1]

    def sample_forward_process(
        self,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the discrete flow matching model, return all the states in the sampling process.
        Arguments:
            t_span (:obj:`torch.Tensor`, optional): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`, optional): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`, optional): The initial state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`, optional): The condition.
            with_grad (:obj:`bool`, optional): Whether to keep the gradient.
            solver_config (:obj:`EasyDict`, optional): The configuration of the solver.
        """
        t_span = torch.linspace(0, self.t_max, 1000) if t_span is None else t_span

        xt = torch.ones(batch_size, self.variable_num) * (self.dimension - 1)
        xt = xt.long()
        xt = xt.to(self.device)

        xt_history = []
        xt_history.append(xt)

        softmax = torch.nn.Softmax(dim=-1)

        for t, t_next in zip(t_span[:-1], t_span[1:]):
            t = t.to(self.device)
            t_next = t_next.to(self.device)
            t = t.repeat(batch_size)
            t_next = t_next.repeat(batch_size)
            probability_denoiser = self.model(t, xt)  # of shape (B, N, D)
            probability_denoiser_softmax = softmax(probability_denoiser)
            xt_one_hot = torch.nn.functional.one_hot(
                xt.long(), num_classes=self.dimension
            ).float()  # Shape: (B, N, D)
            conditional_probability_velocity = torch.einsum(
                "b,bij->bij", 1 / (1 - t), probability_denoiser_softmax - xt_one_hot
            )
            xt_new = xt_one_hot + torch.einsum(
                "b,bij->bij", t_next - t, conditional_probability_velocity
            )
            # sample from xt_new
            xt = torch.distributions.Categorical(probs=xt_new).sample()
            xt_history.append(xt)

        xt = torch.stack(xt_history, dim=0)

        return xt

    def flow_matching_loss(
        self,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            The loss function for the discrete flow matching model.
        Arguments:
            x0 (:obj:`torch.Tensor`): The initial state.
            x1 (:obj:`torch.Tensor`): The final state.
            condition (:obj:`torch.Tensor`, optional): The condition.
            average (:obj:`bool`, optional): Whether to average the loss.
        Returns:
            loss (:obj:`torch.Tensor`): The loss.

            .. math::
                loss = - \mathbb{E}_{t,(X_0,X_1),X_t} p_t(z|x_0, x_1, theta)

        Shapes:
            - x0 (:obj:`torch.Tensor`): :math:`(B, N)`
            - x1 (:obj:`torch.Tensor`): :math:`(B, N)`
            - condition (:obj:`torch.Tensor`, optional): :math:`(B, N)`
            - loss (:obj:`torch.Tensor`): :math:`(B,)`

        """

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

        # Get the random time t
        batch_size, device = get_batch_size_and_device(x0)

        t_random = torch.rand(batch_size, device=device) * self.t_max

        wt_xt_condition_x0_x1 = self.scheduler.pt_z_condition_x0_x1(t_random, x0, x1)
        # wt_xt_condition_x0_x1 is of shape (B, N, D)

        # get xt of shape (B,N) sampled from wt_xt_condition_x0_x1 of shape (B, N, D)
        xt = torch.distributions.Categorical(probs=wt_xt_condition_x0_x1).sample()

        # get the probability of yt given xt and t, which is of shape (B, N, D)
        probability_denoiser = self.model(t_random, xt, condition)

        # calclulate w_y_condition_x0_x1
        x1_one_hot = torch.nn.functional.one_hot(
            x1.long(), num_classes=self.dimension
        ).float()  # Shape: (B, N, D)

        softmax = torch.nn.Softmax(dim=-1)
        probability_denoiser_softmax = softmax(probability_denoiser)
        # probability_denoiser_softmax is of shape (B, N, D)

        eps = 1e-6
        probability_denoiser_softmax = torch.clamp(
            probability_denoiser_softmax, eps, 1 - eps
        )

        loss = -torch.sum(
            x1_one_hot * torch.log(probability_denoiser_softmax), dim=[-1, -2]
        )

        if torch.any(torch.isnan(loss)):
            print("loss is nan")

        # drop item if it is nan
        loss = loss[~torch.isnan(loss)]

        return loss.mean() if average else loss
