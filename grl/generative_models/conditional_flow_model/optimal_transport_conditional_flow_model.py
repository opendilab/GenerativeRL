from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import ot
import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from grl.generative_models.intrinsic_model import IntrinsicModel

from grl.generative_models.model_functions.velocity_function import VelocityFunction
from grl.generative_models.random_generator import gaussian_random_variable
from grl.generative_models.stochastic_process import StochasticProcess
from grl.numerical_methods.numerical_solvers import get_solver
from grl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from grl.numerical_methods.numerical_solvers.ode_solver import (
    DictTensorODESolver,
    ODESolver,
)
from grl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from grl.numerical_methods.probability_path import (
    ConditionalProbabilityPath,
)
from grl.utils import find_parameters


class OptimalTransportConditionalFlowModel(nn.Module):
    """
    Overview:
        The optimal transport conditional flow model, which is based on an optimal transport plan between two distributions.
    Interfaces:
        ``__init__``, ``get_type``, ``sample``, ``sample_forward_process``, ``flow_matching_loss``
    """

    def __init__(
        self,
        config: EasyDict,
    ):
        """
        Overview:
            Initialize the model.
        Arguments:
            - config (:obj:`EasyDict`): The configuration of the model.
        """
        super().__init__()
        self.config = config

        self.x_size = config.x_size
        self.device = config.device

        self.gaussian_generator = gaussian_random_variable(
            config.x_size,
            config.device,
            config.use_tree_tensor if hasattr(config, "use_tree_tensor") else False,
        )

        self.path = ConditionalProbabilityPath(config.path)
        if hasattr(config, "reverse_path"):
            self.reverse_path = ConditionalProbabilityPath(config.reverse_path)
        else:
            self.reverse_path = None
        self.model_type = config.model.type
        assert self.model_type in [
            "velocity_function",
        ], "Unknown type of model {}".format(self.model_type)
        self.model = IntrinsicModel(config.model.args)
        self.diffusion_process = StochasticProcess(self.path)
        self.velocity_function_ = VelocityFunction(
            self.model_type, self.diffusion_process
        )

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "OptimalTransportConditionalFlowModel"

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
            Sample from the model, return the final state.

        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            t_span: :math:`(T)`, where :math:`T` is the number of time steps.
            batch_size: :math:`(B)`, where :math:`B` is the batch size of data, which could be a scalar or a tensor such as :math:`(B1, B2)`.
            x_0: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            condition: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the condition, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x: :math:`(N, D)`, if extra batch size :math:`B` is provided, the shape will be :math:`(B, N, D)`. If x_0 is not provided, the shape will be :math:`(B, D)`. If x_0 and condition are not provided, the shape will be :math:`(D)`.
        """

        return self.sample_forward_process(
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
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
            Sample from the model, return all intermediate states.

        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): An extra batch size used for repeated sampling with the same initial state.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            t_span: :math:`(T)`, where :math:`T` is the number of time steps.
            batch_size: :math:`(B)`, where :math:`B` is the batch size of data, which could be a scalar or a tensor such as :math:`(B1, B2)`.
            x_0: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            condition: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the condition, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x: :math:`(T, N, D)`, if extra batch size :math:`B` is provided, the shape will be :math:`(T, B, N, D)`. If x_0 is not provided, the shape will be :math:`(T, B, D)`. If x_0 and condition are not provided, the shape will be :math:`(T, D)`.
        """

        if t_span is not None:
            t_span = t_span.to(self.device)

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            data_batch_size = x_0.shape[0]
        elif x_0 is not None:
            data_batch_size = x_0.shape[0]
        elif condition is not None:
            data_batch_size = condition.shape[0]
        else:
            data_batch_size = 1

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        if x_0 is None:
            x = self.gaussian_generator(
                batch_size=torch.prod(extra_batch_size) * data_batch_size
            )
            # x.shape = (B*N, D)
        else:
            if isinstance(self.x_size, int):
                assert (
                    torch.Size([self.x_size]) == x_0[0].shape
                ), "The shape of x_0 must be the same as the x_size that is specified in the config"
            elif (
                isinstance(self.x_size, Tuple)
                or isinstance(self.x_size, List)
                or isinstance(self.x_size, torch.Size)
            ):
                assert (
                    torch.Size(self.x_size) == x_0[0].shape
                ), "The shape of x_0 must be the same as the x_size that is specified in the config"
            else:
                assert False, "Invalid x_size"

            x = torch.repeat_interleave(x_0, torch.prod(extra_batch_size), dim=0)
            # x.shape = (B*N, D)

        if condition is not None:
            if isinstance(condition, torch.Tensor):
                condition = torch.repeat_interleave(
                    condition, torch.prod(extra_batch_size), dim=0
                )
                # condition.shape = (B*N, D)
            elif isinstance(condition, TensorDict):
                condition = TensorDict(
                    {
                        key: torch.repeat_interleave(
                            condition[key], torch.prod(extra_batch_size), dim=0
                        )
                        for key in condition.keys()
                    },
                    batch_size=torch.prod(extra_batch_size) * condition.shape,
                    device=condition.device,
                )
            else:
                raise NotImplementedError("Not implemented")

        if isinstance(solver, DPMSolver):
            raise NotImplementedError("Not implemented")
        elif isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            def drift(t, x):
                return self.model(t=t, x=x, condition=condition)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self.model),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                        )
        elif isinstance(solver, DictTensorODESolver):
            # TODO: make it compatible with TensorDict
            def drift(t, x):
                return self.model(t=t, x=x, condition=condition)

            if with_grad:
                data = solver.integrate(
                    drift=drift,
                    x0=x,
                    t_span=t_span,
                    batch_size=torch.prod(extra_batch_size) * data_batch_size,
                    x_size=x.shape,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        batch_size=torch.prod(extra_batch_size) * data_batch_size,
                        x_size=x.shape,
                    )
        elif isinstance(solver, SDESolver):
            raise NotImplementedError("Not implemented")
        else:
            raise NotImplementedError(
                "Solver type {} is not implemented".format(self.config.solver.type)
            )

        if isinstance(data, torch.Tensor):
            # data.shape = (T, B*N, D)
            if len(extra_batch_size.shape) == 0:
                if isinstance(self.x_size, int):
                    data = data.reshape(
                        -1, extra_batch_size, data_batch_size, self.x_size
                    )
                elif (
                    isinstance(self.x_size, Tuple)
                    or isinstance(self.x_size, List)
                    or isinstance(self.x_size, torch.Size)
                ):
                    data = data.reshape(
                        -1, extra_batch_size, data_batch_size, *self.x_size
                    )
                else:
                    assert False, "Invalid x_size"
            else:
                if isinstance(self.x_size, int):
                    data = data.reshape(
                        -1, *extra_batch_size, data_batch_size, self.x_size
                    )
                elif (
                    isinstance(self.x_size, Tuple)
                    or isinstance(self.x_size, List)
                    or isinstance(self.x_size, torch.Size)
                ):
                    data = data.reshape(
                        -1, *extra_batch_size, data_batch_size, *self.x_size
                    )
                else:
                    assert False, "Invalid x_size"
            # data.shape = (T, B, N, D)

            if batch_size is None:
                if x_0 is None and condition is None:
                    data = data.squeeze(1).squeeze(1)
                    # data.shape = (T, D)
                else:
                    data = data.squeeze(1)
                    # data.shape = (T, N, D)
            else:
                if x_0 is None and condition is None:
                    data = data.squeeze(1 + len(extra_batch_size.shape))
                    # data.shape = (T, B, D)
                else:
                    # data.shape = (T, B, N, D)
                    pass
        elif isinstance(data, TensorDict):
            raise NotImplementedError("Not implemented")
        elif isinstance(data, treetensor.torch.Tensor):
            for key in data.keys():
                if len(extra_batch_size.shape) == 0:
                    if isinstance(self.x_size[key], int):
                        data[key] = data[key].reshape(
                            -1, extra_batch_size, data_batch_size, self.x_size[key]
                        )
                    elif (
                        isinstance(self.x_size[key], Tuple)
                        or isinstance(self.x_size[key], List)
                        or isinstance(self.x_size[key], torch.Size)
                    ):
                        data[key] = data[key].reshape(
                            -1, extra_batch_size, data_batch_size, *self.x_size[key]
                        )
                    else:
                        assert False, "Invalid x_size"
                else:
                    if isinstance(self.x_size[key], int):
                        data[key] = data[key].reshape(
                            -1, *extra_batch_size, data_batch_size, self.x_size[key]
                        )
                    elif (
                        isinstance(self.x_size[key], Tuple)
                        or isinstance(self.x_size[key], List)
                        or isinstance(self.x_size[key], torch.Size)
                    ):
                        data[key] = data[key].reshape(
                            -1, *extra_batch_size, data_batch_size, *self.x_size[key]
                        )
                    else:
                        assert False, "Invalid x_size"
                # data.shape = (T, B, N, D)

                if batch_size is None:
                    if x_0 is None and condition is None:
                        data[key] = data[key].squeeze(1).squeeze(1)
                        # data.shape = (T, D)
                    else:
                        data[key] = data[key].squeeze(1)
                        # data.shape = (T, N, D)
                else:
                    if x_0 is None and condition is None:
                        data[key] = data[key].squeeze(1 + len(extra_batch_size.shape))
                        # data.shape = (T, B, D)
                    else:
                        # data.shape = (T, B, N, D)
                        pass
        else:
            raise NotImplementedError("Not implemented")

        return data

    def flow_matching_loss_small_batch_OT_plan(
        self,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition, using the optimal transport plan for small batch size to accelerate the computation.
        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if condition is not None:

            split_size = 128
            x0_split = torch.split(x0, split_size, dim=0)
            x1_split = torch.split(x1, split_size, dim=0)

            x0_ot = []
            x1_ot = []

            condition_split = torch.split(condition, split_size, dim=0)
            condition_ot = []

            for x0_i, x1_i, condition_i in zip(x0_split, x1_split, condition_split):

                a = ot.unif(x0_i.shape[0])
                b = ot.unif(x1_i.shape[0])
                # TODO: make it compatible with TensorDict and treetensor.torch.Tensor
                if x0_i.dim() > 2:
                    x0_i_ = x0_i.reshape(x0_i.shape[0], -1)
                else:
                    x0_i_ = x0_i
                if x1_i.dim() > 2:
                    x1_i_ = x1_i.reshape(x1_i.shape[0], -1)
                else:
                    x1_i_ = x1_i

                M = torch.cdist(x0_i_, x1_i_) ** 2
                p = ot.emd(a, b, M.detach().cpu().numpy())
                assert np.all(np.isfinite(p)), "p is not finite"

                p_flatten = p.flatten()
                p_flatten = p_flatten / p_flatten.sum()

                choices = np.random.choice(
                    p.shape[0] * p.shape[1],
                    p=p_flatten,
                    size=x0_i.shape[0],
                    replace=True,
                )

                i, j = np.divmod(choices, p.shape[1])
                x0_ot_i = x0_i[i]
                x0_ot.append(x0_ot_i)
                x1_ot_i = x1_i[j]
                x1_ot.append(x1_ot_i)

                # condition_ot = condition0_ot = condition1_ot = condition[j]
                condition_ot_i = condition_i[j]
                condition_ot.append(condition_ot_i)

            # torch stack
            x0_ot = torch.stack(x0_ot).reshape(x0.shape)
            x1_ot = torch.stack(x1_ot).reshape(x1.shape)
            condition_ot = torch.stack(condition_ot).reshape(condition.shape)

            return self.velocity_function_.flow_matching_loss_icfm(
                self.model, x0_ot, x1_ot, condition_ot, average
            )

        else:

            split_size = 128

            x0_split = torch.split(x0, split_size, dim=0)
            x1_split = torch.split(x1, split_size, dim=0)

            x0_ot = []
            x1_ot = []

            for x0_i, x1_i in zip(x0_split, x1_split):

                a = ot.unif(x0_i.shape[0])
                b = ot.unif(x1_i.shape[0])
                # TODO: make it compatible with TensorDict and treetensor.torch.Tensor
                if x0_i.dim() > 2:
                    x0_i_ = x0_i.reshape(x0_i.shape[0], -1)
                else:
                    x0_i_ = x0_i
                if x1_i.dim() > 2:
                    x1_i_ = x1_i.reshape(x1_i.shape[0], -1)
                else:
                    x1_i_ = x1_i

                M = torch.cdist(x0_i_, x1_i_) ** 2
                p = ot.emd(a, b, M.detach().cpu().numpy())
                assert np.all(np.isfinite(p)), "p is not finite"

                p_flatten = p.flatten()
                p_flatten = p_flatten / p_flatten.sum()

                choices = np.random.choice(
                    p.shape[0] * p.shape[1],
                    p=p_flatten,
                    size=x0_i.shape[0],
                    replace=True,
                )

                i, j = np.divmod(choices, p.shape[1])
                x0_ot_i = x0_i[i]
                x0_ot.append(x0_ot_i)
                x1_ot_i = x1_i[j]
                x1_ot.append(x1_ot_i)

            # torch stack
            x0_ot = torch.stack(x0_ot, dim=0).reshape(x0.shape)
            x1_ot = torch.stack(x1_ot, dim=0).reshape(x1.shape)

            return self.velocity_function_.flow_matching_loss_icfm(
                self.model, x0_ot, x1_ot, condition, average
            )

    def flow_matching_loss(
        self,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition, using the optimal transport plan to match samples from two distributions.
        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        a = ot.unif(x0.shape[0])
        b = ot.unif(x1.shape[0])
        # TODO: make it compatible with TensorDict and treetensor.torch.Tensor
        if x0.dim() > 2:
            x0_ = x0.reshape(x0.shape[0], -1)
        else:
            x0_ = x0
        if x1.dim() > 2:
            x1_ = x1.reshape(x1.shape[0], -1)
        else:
            x1_ = x1

        M = torch.cdist(x0_, x1_) ** 2
        p = ot.emd(a, b, M.detach().cpu().numpy())
        assert np.all(np.isfinite(p)), "p is not finite"

        p_flatten = p.flatten()
        p_flatten = p_flatten / p_flatten.sum()

        choices = np.random.choice(
            p.shape[0] * p.shape[1], p=p_flatten, size=x0.shape[0], replace=True
        )

        i, j = np.divmod(choices, p.shape[1])
        x0_ot = x0[i]
        x1_ot = x1[j]
        if condition is not None:
            # condition_ot = condition0_ot = condition1_ot = condition[j]
            condition_ot = condition[j]
        else:
            condition_ot = None

        return self.velocity_function_.flow_matching_loss_icfm(
            self.model, x0_ot, x1_ot, condition_ot, average
        )
