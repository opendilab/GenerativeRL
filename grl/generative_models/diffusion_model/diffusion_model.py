from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from grl.generative_models.diffusion_process import DiffusionProcess
from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.model_functions.data_prediction_function import (
    DataPredictionFunction,
)
from grl.generative_models.model_functions.noise_function import NoiseFunction
from grl.generative_models.model_functions.score_function import ScoreFunction
from grl.generative_models.model_functions.velocity_function import VelocityFunction
from grl.generative_models.random_generator import gaussian_random_variable
from grl.generative_models.metric import compute_likelihood
from grl.numerical_methods.numerical_solvers import get_solver
from grl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from grl.numerical_methods.numerical_solvers.ode_solver import (
    DictTensorODESolver,
    ODESolver,
)
from grl.numerical_methods.numerical_solvers.sde_solver import SDESolver
from grl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from grl.utils import find_parameters


class DiffusionModel(nn.Module):
    """
    Overview:
        General diffusion model class that supports various types of continuous-time diffusion paths, which supports sampling, computing score function and velocity function.
        It can be modeled via score function, noise function, velocity function, or data prediction function.
        Both score matching loss and flow matching loss are supported.
    Interfaces:
        ``__init__``, ``sample``, ``score_function``, ``score_matching_loss``, ``velocity_function``, ``flow_matching_loss``.
    """

    def __init__(self, config: EasyDict) -> None:
        """
        Overview:
            Initialization of Diffusion Model.

        Arguments:
            config (:obj:`EasyDict`): The configuration.
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

        self.path = GaussianConditionalProbabilityPath(config.path)
        if hasattr(config, "reverse_path"):
            self.reverse_path = GaussianConditionalProbabilityPath(config.reverse_path)
        else:
            self.reverse_path = None
        self.model_type = config.model.type
        assert self.model_type in [
            "score_function",
            "data_prediction_function",
            "velocity_function",
            "noise_function",
        ], "Unknown type of model {}".format(self.model_type)
        self.model = IntrinsicModel(config.model.args)
        self.diffusion_process = DiffusionProcess(self.path)
        if self.reverse_path is not None:
            self.reverse_diffusion_process = DiffusionProcess(self.reverse_path)
        else:
            self.reverse_diffusion_process = None
        self.score_function_ = ScoreFunction(self.model_type, self.diffusion_process)
        self.velocity_function_ = VelocityFunction(
            self.model_type, self.diffusion_process
        )
        self.noise_function_ = NoiseFunction(self.model_type, self.diffusion_process)
        self.data_prediction_function_ = DataPredictionFunction(
            self.model_type, self.diffusion_process
        )

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "DiffusionModel"

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
            Sample from the diffusion model, return the final state.

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
            if type(x_0) == type(condition):
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
            elif isinstance(condition, treetensor.torch.Tensor):
                for key in condition.keys():
                    condition[key] = torch.repeat_interleave(
                        condition[key], torch.prod(extra_batch_size), dim=0
                    )
                # condition.shape = (B*N, D)   
            elif isinstance(condition, TensorDict):
                for key in condition.keys():
                    condition[key] = torch.repeat_interleave(
                        condition[key], torch.prod(extra_batch_size), dim=0
                    )
            else:
                raise NotImplementedError("Not implemented")


        if isinstance(solver, DPMSolver):
            # Note: DPMSolver does not support t_span argument assignment
            assert (
                t_span is None
            ), "DPMSolver does not support t_span argument assignment"
            # TODO: make it compatible with TensorDict
            if with_grad:
                data = solver.integrate(
                    diffusion_process=self.diffusion_process,
                    noise_function=self.noise_function,
                    data_prediction_function=self.data_prediction_function,
                    x=x,
                    condition=condition,
                    save_intermediate=True,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        diffusion_process=self.diffusion_process,
                        noise_function=self.noise_function,
                        data_prediction_function=self.data_prediction_function,
                        x=x,
                        condition=condition,
                        save_intermediate=True,
                    )
        elif isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(
                        function=self.model,
                        function_type=self.model_type,
                        condition=condition,
                    ).drift,
                    x0=x,
                    t_span=t_span,
                    adjoint_params=find_parameters(self.model),
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(
                            function=self.model,
                            function_type=self.model_type,
                            condition=condition,
                        ).drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
        elif isinstance(solver, DictTensorODESolver):
            # TODO: make it compatible with TensorDict
            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.reverse_ode(
                        function=self.model,
                        function_type=self.model_type,
                        condition=condition,
                    ).drift,
                    x0=x,
                    t_span=t_span,
                    batch_size=torch.prod(extra_batch_size) * data_batch_size,
                    x_size=x.shape,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.reverse_ode(
                            function=self.model,
                            function_type=self.model_type,
                            condition=condition,
                        ).drift,
                        x0=x,
                        t_span=t_span,
                        batch_size=torch.prod(extra_batch_size) * data_batch_size,
                        x_size=x.shape,
                    )
        elif isinstance(solver, SDESolver):
            # TODO: make it compatible with TensorDict
            # TODO: validate the implementation
            assert (
                self.reverse_diffusion_process is not None
            ), "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(
                    self.device
                )
            sde = self.diffusion_process.reverse_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                reverse_diffusion_function=self.reverse_diffusion_process.diffusion,
                reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared,
            )
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=t_span,
                    )
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

    def sample_with_fixed_x(
        self,
        fixed_x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        fixed_mask: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the diffusion model with fixed x, return the final state.

        Arguments:
            fixed_x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The fixed x.
            fixed_mask (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The fixed mask.
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            fixed_x: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            fixed_mask: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the mask, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            t_span: :math:`(T)`, where :math:`T` is the number of time steps.
            batch_size: :math:`(B)`, where :math:`B` is the batch size of data, which could be a scalar or a tensor such as :math:`(B1, B2)`.
            x_0: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            condition: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the condition, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            x: :math:`(N, D)`, if extra batch size :math:`B` is provided, the shape will be :math:`(B, N, D)`. If x_0 is not provided, the shape will be :math:`(B, D)`. If x_0 and condition are not provided, the shape will be :math:`(D,)`.
        """

        return self.sample_forward_process_with_fixed_x(
            fixed_x=fixed_x,
            fixed_mask=fixed_mask,
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

    def sample_forward_process_with_fixed_x(
        self,
        fixed_x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        fixed_mask: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the diffusion model with fixed x, return all intermediate states.

        Arguments:
            fixed_x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The fixed x.
            fixed_mask (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The fixed mask.
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            x_0 (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The initial state, if not provided, it will be sampled from the Gaussian distribution.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.

        Returns:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The sampled result.

        Shapes:
            fixed_x: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the state, which could be a scalar or a tensor such as :math:`(D1, D2)`.
            fixed_mask: :math:`(N, D)`, where :math:`N` is the batch size of data and :math:`D` is the dimension of the mask, which could be a scalar or a tensor such as :math:`(D1, D2)`.
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

        data_batch_size = fixed_x.shape[0]
        assert (
            fixed_x.shape[0] == fixed_mask.shape[0]
        ), "The batch size of fixed_x and fixed_mask must be the same"
        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            assert (
                x_0.shape[0] == fixed_x.shape[0]
            ), "The batch size of x_0 and fixed_x must be the same"
        elif x_0 is not None:
            assert (
                x_0.shape[0] == fixed_x.shape[0]
            ), "The batch size of x_0 and fixed_x must be the same"
        elif condition is not None:
            assert (
                condition.shape[0] == fixed_x.shape[0]
            ), "The batch size of condition and fixed_x must be the same"
        else:
            pass

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
            condition = torch.repeat_interleave(
                condition, torch.prod(extra_batch_size), dim=0
            )
            # condition.shape = (B*N, D)

        fixed_x = torch.repeat_interleave(fixed_x, torch.prod(extra_batch_size), dim=0)
        fixed_mask = torch.repeat_interleave(
            fixed_mask, torch.prod(extra_batch_size), dim=0
        )

        if isinstance(solver, DPMSolver):
            # TODO: make it compatible with DPM solver
            assert False, "Not implemented"
        elif isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict

            x = fixed_x * (1 - fixed_mask) + x * fixed_mask

            def drift_fixed_x(t, x):
                xt_partially_fixed = (
                    self.diffusion_process.direct_sample(
                        self.diffusion_process.t_max - t, fixed_x
                    )
                    * (1 - fixed_mask)
                    + x * fixed_mask
                )
                return fixed_mask * self.diffusion_process.reverse_ode(
                    function=self.model,
                    function_type=self.model_type,
                    condition=condition,
                ).drift(t, xt_partially_fixed)

            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    x0=x,
                    t_span=t_span,
                    adjoint_params=find_parameters(self.model),
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
        elif isinstance(solver, DictTensorODESolver):
            # TODO: make it compatible with TensorDict

            x = fixed_x * (1 - fixed_mask) + x * fixed_mask

            def drift_fixed_x(t, x):
                xt_partially_fixed = (
                    self.diffusion_process.direct_sample(
                        self.diffusion_process.t_max - t, fixed_x
                    )
                    * (1 - fixed_mask)
                    + x * fixed_mask
                )
                return fixed_mask * self.diffusion_process.reverse_ode(
                    function=self.model,
                    function_type=self.model_type,
                    condition=condition,
                ).drift(t, xt_partially_fixed)

            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    x0=x,
                    t_span=t_span,
                    batch_size=torch.prod(extra_batch_size) * data_batch_size,
                    x_size=x.shape,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        x0=x,
                        t_span=t_span,
                        batch_size=torch.prod(extra_batch_size) * data_batch_size,
                        x_size=x.shape,
                    )
        elif isinstance(solver, SDESolver):
            # TODO: make it compatible with TensorDict
            # TODO: validate the implementation
            assert (
                self.reverse_diffusion_process is not None
            ), "reverse_path must be specified in config"
            if not hasattr(self, "t_span"):
                self.t_span = torch.linspace(0, self.diffusion_process.t_max, 2).to(
                    self.device
                )

            x = fixed_x * (1 - fixed_mask) + x * fixed_mask
            sde = self.diffusion_process.reverse_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                reverse_diffusion_function=self.reverse_diffusion_process.diffusion,
                reverse_diffusion_squared_function=self.reverse_diffusion_process.diffusion_squared,
            )

            def drift_fixed_x(t, x):
                xt_partially_fixed = (
                    self.diffusion_process.direct_sample(
                        self.diffusion_process.t_max - t, fixed_x
                    )
                    * (1 - fixed_mask)
                    + x * fixed_mask
                )
                return fixed_mask * sde.drift(t, xt_partially_fixed)

            def diffusion_fixed_x(t, x):
                xt_partially_fixed = (
                    self.diffusion_process.direct_sample(
                        self.diffusion_process.t_max - t, fixed_x
                    )
                    * (1 - fixed_mask)
                    + x * fixed_mask
                )
                return fixed_mask * sde.diffusion(t, xt_partially_fixed)

            if with_grad:
                data = solver.integrate(
                    drift=drift_fixed_x,
                    diffusion=diffusion_fixed_x,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=drift_fixed_x,
                        diffusion=diffusion_fixed_x,
                        x0=x,
                        t_span=t_span,
                    )
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
                data = data.squeeze(1)
                # data.shape = (T, N, D)
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
                    data[key] = data[key].squeeze(1)
                    # data.shape = (T, N, D)
                else:
                    # data.shape = (T, B, N, D)
                    pass
        else:
            raise NotImplementedError("Not implemented")

        return data

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
            Use forward path of the diffusion model given the sampled x. Note that this is not the reverse process, and thus is not designed for sampling form the diffusion model.
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
            # TODO: make it compatible with TensorDict

            if with_grad:
                data = solver.integrate(
                    drift=self.diffusion_process.forward_ode(
                        function=self.model,
                        function_type=self.model_type,
                        condition=condition,
                    ).drift,
                    x0=x,
                    t_span=t_span,
                    adjoint_params=find_parameters(self.model),
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=self.diffusion_process.forward_ode(
                            function=self.model,
                            function_type=self.model_type,
                            condition=condition,
                        ).drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
        elif isinstance(solver, SDESolver):
            # TODO: make it compatible with TensorDict
            # TODO: validate the implementation
            assert (
                self.diffusion_process is not None
            ), "path must be specified in config"

            sde = self.diffusion_process.forward_sde(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
                forward_diffusion_function=self.diffusion_process.diffusion,
                forward_diffusion_squared_function=self.diffusion_process.diffusion_squared,
            )
            if with_grad:
                data = solver.integrate(
                    drift=sde.drift,
                    diffusion=sde.diffusion,
                    x0=x,
                    t_span=t_span,
                )
            else:
                with torch.no_grad():
                    data = solver.integrate(
                        drift=sde.drift,
                        diffusion=sde.diffusion,
                        x0=x,
                        t_span=t_span,
                    )
        else:
            raise NotImplementedError(
                "Solver type {} is not implemented".format(self.config.solver.type)
            )
        return data

    def score_function(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        r"""
        Overview:
            Return score function of the model at time t given the initial state, which is the gradient of the log-likelihood.

            .. math::
                \nabla_{x_t} \log p_{\theta}(x_t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        return self.score_function_.forward(self.model, t, x, condition)

    def score_matching_loss(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        weighting_scheme: str = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            The loss function for training unconditional diffusion model.

        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            weighting_scheme (:obj:`str`): The weighting scheme for score matching loss, which can be "maximum_likelihood" or "vanilla".

            ..note::
                - "maximum_likelihood": The weighting scheme is based on the maximum likelihood estimation. Refer to the paper "Maximum Likelihood Training of Score-Based Diffusion Models" for more details. The weight :math:`\lambda(t)` is denoted as:

                    .. math::
                        \lambda(t) = g^2(t)

                    for numerical stability, we use Monte Carlo sampling to approximate the integral of :math:`\lambda(t)`.

                    .. math::
                        \lambda(t) = g^2(t) = p(t)\sigma^2(t)

                - "vanilla": The weighting scheme is based on the vanilla score matching, which balances the MSE loss by scaling the model output to the noise value. Refer to the paper "Score-Based Generative Modeling through Stochastic Differential Equations" for more details. The weight :math:`\lambda(t)` is denoted as:

                    .. math::
                        \lambda(t) = \sigma^2(t)
        """

        return self.score_function_.score_matching_loss(
            self.model, x, condition, self.gaussian_generator, weighting_scheme, average
        )

    def velocity_function(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        r"""
        Overview:
            Return velocity of the model at time t given the initial state.

            .. math::
                v_{\theta}(t, x)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state at time t.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        return self.velocity_function_.forward(self.model, t, x, condition)

    def flow_matching_loss(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss function of the model given the initial state and the condition.
        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            average (:obj:`bool`): Whether to average the loss across the batch.
        """

        return self.velocity_function_.flow_matching_loss(
            self.model, x, condition, self.gaussian_generator, average
        )

    def noise_function(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        r"""
        Overview:
            Return noise function of the model at time t given the initial state.

            .. math::
                - \sigma(t) \nabla_{x_t} \log p_{\theta}(x_t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        return self.noise_function_.forward(self.model, t, x, condition)

    def data_prediction_function(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        r"""
        Overview:
            Return data prediction function of the model at time t given the initial state.

            .. math::
                \frac{- \sigma(t) x_t + \sigma^2(t) \nabla_{x_t} \log p_{\theta}(x_t)}{s(t)}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        return self.data_prediction_function_.forward(self.model, t, x, condition)

    def log_prob(
        self,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        using_Hutchinson_trace_estimator: bool = True,
        with_grad: bool = False
    ):
        r"""
        Overview:
            Return the log probability of the model given the initial state and the condition.

            .. math::
                \log p_{\theta}(x)

        Arguments:
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
        """

        if with_grad:
            return compute_likelihood(
                model=self, x=x, condition=condition, using_Hutchinson_trace_estimator=using_Hutchinson_trace_estimator
            )
        else:
            with torch.no_grad():
                return compute_likelihood(
                    model=self, x=x, condition=condition, using_Hutchinson_trace_estimator=using_Hutchinson_trace_estimator
                )

    def sample_with_log_prob(
        self,
        t_span: torch.Tensor,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        r"""
        Overview:
            Sample from the model and return the log probability of the sampled result.

        Arguments:
            t_span (:obj:`torch.Tensor`): The time span.
            batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            with_grad (:obj:`bool`): Whether to return the gradient.
            solver_config (:obj:`EasyDict`): The configuration of the solver.
        """

        x = self.sample_forward_process(
            t_span=t_span,
            batch_size=batch_size,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

        return x, self.log_prob(x=x, condition=condition, with_grad=with_grad)

    def dpo_loss(
        self,
        ref_dm,
        data,
        beta,
    ) -> torch.Tensor:
        """
        Overview:
            The loss function for training the diffusion process by Direct Policy Optimization (DPO).
            This is an in-development feature and is not recommended for general use.
        """
        # TODO: split data_w, data_l
        x_w, x_l = data[:, :2], data[:, 2:]
        noise = torch.randn_like(x_w).to(x_w.device)
        eps = 1e-5
        t = (
            torch.rand(x_w.shape[0], device=x_w.device)
            * (self.diffusion_process.t_max - eps)
            + eps
        )
        noisy_x_w = (
            self.diffusion_process.scale(t, x_w) * x_w
            + self.diffusion_process.std(t, x_w) * noise
        )
        noisy_x_w = noisy_x_w.to(t)
        noisy_x_l = (
            self.diffusion_process.scale(t, x_l) * x_l
            + self.diffusion_process.std(t, x_l) * noise
        )
        noisy_x_l = noisy_x_l.to(t)
        model_w_pred = self.model(t, noisy_x_w)
        model_l_pred = self.model(t, noisy_x_l)
        ref_w_pred = ref_dm.model(t, noisy_x_w)
        ref_l_pred = ref_dm.model(t, noisy_x_l)
        model_w_err = (model_w_pred - noise).norm(dim=1, keepdim=True).pow(2)
        model_l_err = (model_l_pred - noise).norm(dim=1, keepdim=True).pow(2)
        ref_w_err = (ref_w_pred - noise).norm(dim=1, keepdim=True).pow(2).detach()
        ref_l_err = (ref_l_pred - noise).norm(dim=1, keepdim=True).pow(2).detach()
        w_diff = model_w_err - ref_w_err
        l_diff = model_l_err - ref_l_err
        inside_term = -1 * beta * (w_diff - l_diff)
        loss = -1 * torch.log(torch.sigmoid(inside_term) + eps).mean()
        return loss
