from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
import treetensor
from easydict import EasyDict
from tensordict import TensorDict
import ot
import numpy as np
from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.model_functions.velocity_function import VelocityFunction
from grl.generative_models.random_generator import gaussian_random_variable
from grl.generative_models.stochastic_process import StochasticProcess
from grl.generative_models.metric import compute_likelihood
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


class IndependentConditionalFlowModel(nn.Module):
    """
    Overview:
        The independent conditional flow model, which is a flow model with independent conditional probability paths.
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
            config (:obj:`EasyDict`): The configuration of the model.
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
        return "IndependentConditionalFlowModel"

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

    def sample_with_mask(
        self,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        mask: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        x_1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the model with masked element, return the final state.

        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            mask (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The mask.
            x_1 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The masked element, same shape as x_1.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            t_span: :math:`(T)`, where :math:`T` is the number of time steps.
            batch_size: :math:`(B)`, where :math:`B` is the batch size of data, which could be a scalar or a tensor such as :math:`(B1, B2)`.
            x_0: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            condition: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the condition, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            mask: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the mask, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x_1: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the masked element, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x: :math:`(N, D)`, if extra batch size :math:`B` is provided, the shape will be :math:`(B, N, D)`. If x_0 is not provided, the shape will be :math:`(B, D)`. If x_0 and condition are not provided, the shape will be :math:`(D)`.
        """

        return self.sample_with_mask_forward_process(
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            mask=mask,
            x_1=x_1,
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
            Sample from the diffusion model, return all intermediate states.

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

    def sample_with_mask_forward_process(
        self,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        mask: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        x_1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the diffusion model, return all intermediate states.

        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): An extra batch size used for repeated sampling with the same initial state.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            mask (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The mask.
            x_1 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The masked element, same shape as x_1.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            t_span: :math:`(T)`, where :math:`T` is the number of time steps.
            batch_size: :math:`(B)`, where :math:`B` is the batch size of data, which could be a scalar or a tensor such as :math:`(B1, B2)`.
            x_0: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            condition: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the condition, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            mask: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the mask, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x_1: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the masked element, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x: :math:`(T, N, D)`, if extra batch size :math:`B` is provided, the shape will be :math:`(T, B, N, D)`. If x_0 is not provided, the shape will be :math:`(T, B, D)`. If x_0 and condition are not provided, the shape will be :math:`(T, D)`.
        """

        if mask is None:
            return self.sample_forward_process(
                t_span=t_span,
                batch_size=batch_size,
                x_0=x_0,
                condition=condition,
                with_grad=with_grad,
                solver_config=solver_config,
            )

        else:

            x_1_sampled = self.sample_forward_process(
                t_span=t_span,
                batch_size=batch_size,
                x_0=x_0,
                condition=condition,
                with_grad=with_grad,
                solver_config=solver_config,
            )

            x_1 = torch.where(mask, x_1, x_1_sampled)

            return x_1

    def log_prob(
        self,
        x_1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        log_prob_x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        function_log_prob_x_0: Union[callable, nn.Module] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        t: torch.Tensor = None,
        using_Hutchinson_trace_estimator: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Compute the log probability of the final state given the initial state and the condition.
        Arguments:
            x_1 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The final state.
            log_prob_x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The log probability of the initial state.
            function_log_prob_x_0 (:obj:`Union[callable, nn.Module]`): The function to compute the log probability of the initial state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The condition.
            t (:obj:`torch.Tensor`): The time span.
            using_Hutchinson_trace_estimator (:obj:`bool`): Whether to use Hutchinson trace estimator. It is an approximation of the trace of the Jacobian of the drift function, \
                which is faster but less accurate. We recommend setting it to True for high dimensional data.
        Returns:
            log_likelihood (:obj:`torch.Tensor`): The log likelihood of the final state given the initial state and the condition.
        """

        model_drift = lambda t, x: -self.model(1 - t, x, condition)
        model_params = find_parameters(self.model)

        def compute_trace_of_jacobian_general(dx, x):
            # Assuming x has shape (B, D1, ..., Dn)
            shape = x.shape[1:]  # get the shape of a single element in the batch
            outputs = torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype
            )  # trace for each batch
            # Iterate through each index in the product of dimensions
            for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
                if len(index.shape) > 0:
                    index = tuple(index)
                else:
                    index = (index,)
                grad_outputs = torch.zeros_like(x)
                grad_outputs[(slice(None), *index)] = (
                    1  # set one at the specific index across all batches
                )
                grads = torch.autograd.grad(
                    outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True
                )[0]
                outputs += grads[(slice(None), *index)]
            return outputs

        def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
            """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

            fn_eps = torch.sum(dx * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
            outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
            return outputs

        def composite_drift(t, x):
            # where x is actually x0_and_diff_logp, (x0, diff_logp), which is a tuple containing x and logp_xt_minus_logp_x0
            with torch.set_grad_enabled(True):
                t = t.detach()
                x_t = x[0].detach()
                logp_xt_minus_logp_x0 = x[1]

                x_t.requires_grad = True
                t.requires_grad = True

                dx = model_drift(t, x_t)
                if using_Hutchinson_trace_estimator:
                    noise = torch.randn_like(x_t, device=x_t.device)
                    logp_drift = -compute_trace_of_jacobian_by_Hutchinson_Skilling(
                        dx, x_t, noise
                    )
                    # logp_drift = - divergence_approx(dx, x_t, noise)
                else:
                    logp_drift = -compute_trace_of_jacobian_general(dx, x_t)

                return dx, logp_drift

        # x.shape = [batch_size, state_dim]
        x1_and_diff_logp = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))

        if t is None:
            t_span = torch.linspace(0.0, 1.0, 1000).to(x.device)
        else:
            t_span = t.to(x_1.device)

        solver = ODESolver(library="torchdiffeq_adjoint")

        x0_and_logpx0 = solver.integrate(
            drift=composite_drift,
            x0=x1_and_diff_logp,
            t_span=t_span,
            adjoint_params=model_params,
        )

        logp_x0_minus_logp_x1 = x0_and_logpx0[1][-1]
        x0 = x0_and_logpx0[0][-1]

        if log_prob_x_0 is not None:
            log_likelihood = log_prob_x_0 - logp_x0_minus_logp_x1
        elif function_log_prob_x_0 is not None:
            log_prob_x_0 = function_log_prob_x_0(x0)
            log_likelihood = log_prob_x_0 - logp_x0_minus_logp_x1
        else:
            x0_1d = x0.reshape(x0.shape[0], -1)
            log_prob_x_0 = Independent(
                Normal(
                    loc=torch.zeros_like(x0_1d, device=x0_1d.device),
                    scale=torch.ones_like(x0_1d, device=x0_1d.device),
                ),
                1,
            ).log_prob(x0_1d)

            log_likelihood = log_prob_x_0 - logp_x0_minus_logp_x1

        return log_likelihood

    def sample_with_log_prob(
        self,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
        using_Hutchinson_trace_estimator: bool = True,
    ) -> Tuple[Union[torch.Tensor, TensorDict, treetensor.torch.Tensor], torch.Tensor]:
        """
        Overview:
            Sample from the model, return the final state and the log probability of the initial state.
        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.
            using_Hutchinson_trace_estimator (:obj:`bool`): Whether to use Hutchinson trace estimator. It is an approximation of the trace of the Jacobian of the drift function, \
                which is faster but less accurate. We recommend setting it to True for high dimensional data.
        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.
            log_prob_x_0 (:obj:`torch.Tensor`): The log probability of the initial state.
        """

        x = self.sample(
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )

        log_prob_x_0 = self.log_prob(
            x_1=x,
            condition=condition,
            t=t_span,
            using_Hutchinson_trace_estimator=using_Hutchinson_trace_estimator,
        )

        return x, log_prob_x_0

    def flow_matching_loss(
        self,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition.
        Arguments:
            x0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state.
            x1 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The final state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The condition for the flow matching loss.
            average (:obj:`bool`): Whether to average the loss across the batch.
        """

        return self.velocity_function_.flow_matching_loss_icfm(
            self.model, x0, x1, condition, average
        )

    def flow_matching_loss_with_mask(
        self,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        mask: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition with a mask.
        Arguments:
            x0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state.
            x1 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The final state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The condition for the flow matching loss.
            mask (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The mask signal for x0, which is either True or False and has same shape as x0, if it is True, the corresponding element in x0 will not be used for the loss computation, and the true value of that element is usually not provided in condition.
            average (:obj:`bool`): Whether to average the loss across the batch.
        """

        if mask is None:
            return self.flow_matching_loss(self.model, x0, x1, condition, average)
        else:
            # loss shape is (B, D)
            loss = self.velocity_function_.flow_matching_loss_icfm(
                self.model, x0, x1, condition, average=False, sum_all_elements=False
            )

            # replace the False elements in mask with 0
            loss = loss.masked_fill(~mask, 0.0)

            # num of elements in mask, sum them batch-wise, there maybe more than 1 dim in mask
            num_elements = mask.sum(dim=tuple(range(1, len(mask.shape))))

            # clamp num_elements to be at least 1
            num_elements = torch.clamp(num_elements, min=1)

            # average the loss across the batch
            loss = loss.sum(dim=tuple(range(1, len(loss.shape)))) / num_elements

            if average:
                loss = loss.mean()

            return loss

    def forward_sample(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t_span: torch.Tensor,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Use forward path of the flow model given the sampled x. Note that this is not the reverse process, and thus is not designed for sampling form the flow model.
            Rather, it is used for encode a sampled x to the latent space.

        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            t_span (:obj:`torch.Tensor`): The time span.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.
        """

        return self.forward_sample_process(
            x=x,
            t_span=t_span,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

    def forward_sample_process(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t_span: torch.Tensor,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Use forward path of the diffusion model given the sampled x. Note that this is not the reverse process, and thus is not designed for sampling form the diffusion model.
            Rather, it is used for encode a sampled x to the latent space. Return all intermediate states.

        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            t_span (:obj:`torch.Tensor`): The time span.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.
        """

        # TODO: very important function
        # TODO: validate these functions
        t_span = t_span.to(self.device)

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        if isinstance(solver, ODESolver):

            def reverse_drift(t, x):
                reverse_t = t_span.max() - t + t_span.min()
                return -self.model(t=reverse_t, x=x, condition=condition)

            # TODO: make it compatible with TensorDict
            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=reverse_drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=reverse_drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self.model),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=reverse_drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=reverse_drift,
                            x0=x,
                            t_span=t_span,
                        )
        else:
            raise NotImplementedError(
                "Solver type {} is not implemented".format(self.config.solver.type)
            )
        return data

    def optimal_transport_flow_matching_loss(
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
