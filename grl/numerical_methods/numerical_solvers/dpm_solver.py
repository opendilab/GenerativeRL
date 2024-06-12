#############################################################
# This DPM-Solver snippet is from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
# wich is based on https://github.com/LuChengTHU/dpm-solver
#############################################################

from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class DPMSolver:
    """
    Overview:
        The DPM-Solver for sampling from the diffusion process.
    Interface:
        ``__init__``, ``integrate``
    """

    def __init__(
        self,
        order: int,
        device: str,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        steps: int = None,
        type: str = "dpm_solver",
        method: str = "singlestep",
        solver_type: str = "dpm_solver",
        skip_type: str = "time_uniform",
        denoise: bool = False,
    ):
        """
        Overview:
            Initialize the DPM-Solver.
        Arguments:
            order (:obj:`int`): The order of the DPM-Solver, which should be 1, 2, or 3.
            device (:obj:`str`): The device for the computation.
            denoise (:obj:`bool`): Whether to denoise at the final step.
            atol (:obj:`float`): The absolute tolerance for the adaptive solver.
            rtol (:obj:`float`): The relative tolerance for the adaptive solver.
            steps (:obj:`int`): The total number of function evaluations (NFE).
            type (:obj:`str`): The type for the DPM-Solver, which should be 'dpm_solver' or 'dpm_solver++'.
            method (:obj:`str`): The method for the DPM-Solver, which should be 'singlestep', 'multistep', 'singlestep_fixed', or 'adaptive'.
            solver_type (:obj:`str`): The type for the high-order solvers, which should be 'dpm_solver' or 'taylor'.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
            skip_type (:obj:`str`): The type for the spacing of the time steps, which should be 'logSNR', 'time_uniform', or 'time_quadratic'.
            denoise (:obj:`bool`): Whether to denoise at the final step.
        """
        self.type = type
        assert self.type in ["dpm_solver", "dpm_solver++"]
        if self.type == "dpm_solver++":
            self.use_dpm_solver_plus_plus = True
        else:
            self.use_dpm_solver_plus_plus = False
        self.atol = atol
        self.rtol = rtol
        self.steps = steps
        self.order = order
        assert self.order in [1, 2, 3]
        self.method = method
        assert self.method in [
            "singlestep",
            "multistep",
            "singlestep_fixed",
            "adaptive",
        ]
        self.solver_type = solver_type
        assert self.solver_type in ["dpm_solver", "taylor"]
        if self.solver_type == "dpm_solver":
            self.default_high_order_solvers = True
        else:
            self.default_high_order_solvers = False
        self.skip_type = skip_type
        assert self.skip_type in ["logSNR", "time_uniform", "time_quadratic"]
        self.denoise = denoise
        self.device = device
        self.nfe = 0

        # TODO: support dynamic thresholding for dpm_solver++

    def integrate(
        self,
        diffusion_process,
        noise_function: Callable,
        data_prediction_function: Callable,
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
        steps: int = None,
        save_intermediate: bool = False,
    ):
        """
        Overview:
            Integrate the diffusion process by the DPM-Solver.
        Arguments:
            diffusion_process (:obj:`DiffusionProcess`): The diffusion process.
            noise_function (:obj:`Callable`): The noise prediction model.
            data_prediction_function (:obj:`Callable`): The data prediction model.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `t_start`.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The condition for the data prediction model.
            steps (:obj:`int`): The total number of function evaluations (NFE).
            save_intermediate (:obj:`bool`): If true, also return the intermediate model values.
        Returns:
            x_end (:obj:`torch.Tensor`): The approximated solution at time `t_end`.
        """

        steps = (
            steps if steps is not None else self.steps if self.steps is not None else 20
        )

        def model_fn(
            x: Union[torch.Tensor, TensorDict], t: torch.Tensor
        ) -> Union[torch.Tensor, TensorDict]:
            """
            Overview:
                Convert the model to the noise prediction model or the data prediction model.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The input tensor.
                t (:obj:`torch.Tensor`): The time tensor.
            """
            if self.use_dpm_solver_plus_plus:
                return data_prediction_function(t, x, condition)
            else:
                return noise_function(t, x, condition)

        def get_time_steps(t_T, t_0, N):
            """
            Overview:
                Compute the intermediate time steps for sampling.

            Arguments:
                t_T (:obj:`float`): The starting time of the sampling (default is T).
                t_0 (:obj:`float`): The ending time of the sampling (default is epsilon).
                N (:obj:`int`): The total number of the spacing of the time steps.
            Returns:
                t (:obj:`torch.Tensor`): A pytorch tensor of the time steps, with the shape (N + 1,).
            """
            if self.skip_type == "logSNR":
                lambda_T = diffusion_process.HalfLogSNR(t_T).to(self.device)
                lambda_0 = diffusion_process.HalfLogSNR(t_0).to(self.device)
                logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1)
                return self.diffusion_process.InverseHalfLogSNR(logSNR_steps)
            elif self.skip_type == "time_uniform":
                return torch.linspace(t_T, t_0, N + 1).to(self.device)
            elif self.skip_type == "time_quadratic":
                t_order = 2
                t = (
                    torch.linspace(
                        t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1
                    )
                    .pow(t_order)
                    .to(self.device)
                )
                return t
            else:
                raise ValueError(
                    "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(
                        self.skip_type
                    )
                )

        def get_orders_for_singlestep_solver(steps: int, order: int) -> List[int]:
            """
            Overview:
                Get the order of each step for sampling by the singlestep DPM-Solver.
            Arguments:
                steps (:obj:`int`): The total number of function evaluations (NFE).
                order (:obj:`int`): The max order for the solver (2 or 3).
            Returns:
                orders (:obj:`List[int]`): A list of the solver order of each step.
                
            .. note::
                We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
                Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
                    If order == 1:
                        We take `steps` of DPM-Solver-1 (i.e. DDIM).
                    If order == 2:
                        Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                        If steps % 2 == 0, we use K steps of DPM-Solver-2.
                        If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
                    If order == 3:
                        Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, \
                            and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                        If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                        If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2
            """
            if order == 3:
                K = steps // 3 + 1
                if steps % 3 == 0:
                    orders = [
                        3,
                    ] * (
                        K - 2
                    ) + [2, 1]
                elif steps % 3 == 1:
                    orders = [
                        3,
                    ] * (
                        K - 1
                    ) + [1]
                else:
                    orders = [
                        3,
                    ] * (
                        K - 1
                    ) + [2]
                return orders
            elif order == 2:
                K = steps // 2
                if steps % 2 == 0:
                    # orders = [2,] * K
                    K = steps // 2 + 1
                    orders = [
                        2,
                    ] * (K - 2) + [
                        1,
                    ] * 2
                else:
                    orders = [
                        2,
                    ] * K + [1]
                return orders
            elif order == 1:
                return [
                    1,
                ] * steps
            else:
                raise ValueError("'order' must be '1' or '2' or '3'.")

        def denoise_fn(
            x: Union[torch.Tensor, TensorDict],
            s: torch.Tensor,
            condition: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
            """
            Overview:
                Denoise at the final step, which is equivalent to solve the ODE \
                from lambda_s to infty by first-order discretization.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The input tensor.
                s (:obj:`torch.Tensor`): The time tensor.
                condition (:obj:`Union[torch.Tensor, TensorDict]`): The condition for the data prediction model.
            Returns:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The denoised output.
            """
            return data_prediction_function(s, x, condition)

        def dpm_solver_first_update(
            x: Union[torch.Tensor, TensorDict],
            s: torch.Tensor,
            t: torch.Tensor,
            model_s: Union[torch.Tensor, TensorDict] = None,
            return_intermediate: bool = False,
        ):
            """
            Overview:
                DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                s (:obj:`torch.Tensor`): The starting time, with the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
                model_s (:obj:`Union[torch.Tensor, TensorDict]`): The model function evaluated at time `s`.
                    If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
                return_intermediate (:obj:`bool`): If true, also return the model value at time `s`.
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            lambda_s = diffusion_process.HalfLogSNR(s, x)
            lambda_t = diffusion_process.HalfLogSNR(t, x)
            h = lambda_t - lambda_s
            log_alpha_s = diffusion_process.log_scale(s, x)
            log_alpha_t = diffusion_process.log_scale(t, x)
            sigma_s = diffusion_process.std(s, x)
            sigma_t = diffusion_process.std(t, x)
            alpha_t = torch.exp(log_alpha_t)

            if self.use_dpm_solver_plus_plus:
                phi_1 = torch.expm1(-h)
                if model_s is None:
                    model_s = model_fn(x, s)
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
                if return_intermediate:
                    return x_t, {"model_s": model_s}
                else:
                    return x_t
            else:
                phi_1 = torch.expm1(h)
                if model_s is None:
                    model_s = model_fn(x, s)
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s
                )
                if return_intermediate:
                    return x_t, {"model_s": model_s}
                else:
                    return x_t

        def singlestep_dpm_solver_second_update(
            x: Union[torch.Tensor, TensorDict],
            s: torch.Tensor,
            t: torch.Tensor,
            r1: float = 0.5,
            model_s: Union[torch.Tensor, TensorDict] = None,
            return_intermediate: bool = False,
        ):
            """
            Overview:
                Singlestep solver DPM-Solver-2 from time `s` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                s (:obj:`torch.Tensor`): The starting time, with the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
                r1 (:obj:`float`): The hyperparameter of the second-order solver.
                model_s (:obj:`Union[torch.Tensor, TensorDict]`): The model function evaluated at time `s`.
                    If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
                return_intermediate (:obj:`bool`): If true, also return the model value at time `s` and `s1` (the intermediate time).
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            if r1 is None:
                r1 = 0.5
            lambda_s = diffusion_process.HalfLogSNR(s, x)
            lambda_t = diffusion_process.HalfLogSNR(t, x)
            h = lambda_t - lambda_s
            lambda_s1 = lambda_s + r1 * h
            s1 = diffusion_process.InverseHalfLogSNR(lambda_s1)[:, 0]
            log_alpha_s = diffusion_process.log_scale(s, x)
            log_alpha_s1 = diffusion_process.log_scale(s1, x)
            log_alpha_t = diffusion_process.log_scale(t, x)
            sigma_s = diffusion_process.std(s, x)
            sigma_s1 = diffusion_process.std(s1, x)
            sigma_t = diffusion_process.std(t, x)
            alpha_s1 = torch.exp(log_alpha_s1)
            alpha_t = torch.exp(log_alpha_t)

            if self.use_dpm_solver_plus_plus:
                phi_11 = torch.expm1(-r1 * h)
                phi_1 = torch.expm1(-h)
                if model_s is None:
                    model_s = model_fn(x, s)
                x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
                model_s1 = model_fn(x_s1, s1)
                if self.default_high_order_solvers:
                    x_t = (
                        sigma_t / sigma_s * x
                        - alpha_t * phi_1 * model_s
                        - (0.5 / r1) * alpha_t * phi_1 * (model_s1 - model_s)
                    )
                else:
                    x_t = (
                        sigma_t / sigma_s * x
                        - alpha_t * phi_1 * model_s
                        + (1.0 / r1)
                        * alpha_t
                        * ((torch.exp(-h) - 1.0) / h + 1.0)
                        * (model_s1 - model_s)
                    )
            else:
                phi_11 = torch.expm1(r1 * h)
                phi_1 = torch.expm1(h)

                if model_s is None:
                    model_s = model_fn(x, s)
                x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s) * x
                    - sigma_s1 * phi_11 * model_s
                )
                model_s1 = model_fn(x_s1, s1)
                if self.default_high_order_solvers:
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - sigma_t * phi_1 * model_s
                        - (0.5 / r1) * sigma_t * phi_1 * (model_s1 - model_s)
                    )
                else:
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - sigma_t * phi_1 * model_s
                        - (1.0 / r1)
                        * sigma_t
                        * ((torch.exp(h) - 1.0) / h - 1.0)
                        * (model_s1 - model_s)
                    )
            if return_intermediate:
                return x_t, {"model_s": model_s, "model_s1": model_s1}
            else:
                return x_t

        def singlestep_dpm_solver_third_update(
            x: Union[torch.Tensor, TensorDict],
            s: torch.Tensor,
            t: torch.Tensor,
            r1: float = 1.0 / 3.0,
            r2: float = 2.0 / 3.0,
            model_s: Union[torch.Tensor, TensorDict] = None,
            model_s1: Union[torch.Tensor, TensorDict] = None,
            return_intermediate: bool = False,
        ):
            """
            Overview:
                Singlestep solver DPM-Solver-3 from time `s` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                s (:obj:`torch.Tensor`): The starting time, with the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
                r1 (:obj:`float`): The hyperparameter of the third-order solver.
                r2 (:obj:`float`): The hyperparameter of the third-order solver.
                model_s (:obj:`Union[torch.Tensor, TensorDict]`): The model function evaluated at time `s`.
                    If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
                model_s1 (:obj:`Union[torch.Tensor, TensorDict]`): The model function evaluated at time `s1` (the intermediate time given by `r1`).
                    If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
                return_intermediate (:obj:`bool`): If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            if r1 is None:
                r1 = 1.0 / 3.0
            if r2 is None:
                r2 = 2.0 / 3.0
            lambda_s = diffusion_process.HalfLogSNR(s, x)
            lambda_t = diffusion_process.HalfLogSNR(t, x)
            h = lambda_t - lambda_s
            lambda_s1 = lambda_s + r1 * h
            lambda_s2 = lambda_s + r2 * h
            s1 = diffusion_process.InverseHalfLogSNR(lambda_s1)[:, 0]
            s2 = diffusion_process.InverseHalfLogSNR(lambda_s2)[:, 0]
            log_alpha_s = diffusion_process.log_scale(s, x)
            log_alpha_s1 = diffusion_process.log_scale(s1, x)
            log_alpha_s2 = diffusion_process.log_scale(s2, x)
            log_alpha_t = diffusion_process.log_scale(t, x)
            sigma_s = diffusion_process.std(s, x)
            sigma_s1 = diffusion_process.std(s1, x)
            sigma_s2 = diffusion_process.std(s2, x)
            sigma_t = diffusion_process.std(t, x)
            alpha_s1 = torch.exp(log_alpha_s1)
            alpha_s2 = torch.exp(log_alpha_s2)
            alpha_t = torch.exp(log_alpha_t)
            if self.use_dpm_solver_plus_plus:
                phi_11 = torch.expm1(-r1 * h)
                phi_12 = torch.expm1(-r2 * h)
                phi_1 = torch.expm1(-h)
                phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
                phi_2 = phi_1 / h + 1.0
                phi_3 = phi_2 / h - 0.5
                if model_s is None:
                    model_s = model_fn(x, s)
                if model_s1 is None:
                    x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
                    model_s1 = model_fn(x_s1, s1)
                x_s2 = (
                    sigma_s2 / sigma_s * x
                    - alpha_s2 * phi_12 * model_s
                    + r2 / r1 * alpha_s2 * phi_22 * (model_s1 - model_s)
                )
                model_s2 = model_fn(x_s2, s2)
                if self.default_high_order_solvers:
                    x_t = (
                        sigma_t / sigma_s * x
                        - alpha_t * phi_1 * model_s
                        + (1.0 / r2) * alpha_t * phi_2 * (model_s2 - model_s)
                    )
                else:
                    D1_0 = (1.0 / r1) * (model_s1 - model_s)
                    D1_1 = (1.0 / r2) * (model_s2 - model_s)
                    D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                    D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                    x_t = (
                        sigma_t / sigma_s * x
                        - alpha_t * phi_1 * model_s
                        + alpha_t * phi_2 * D1
                        - alpha_t * phi_3 * D2
                    )
            else:
                phi_11 = torch.expm1(r1 * h)
                phi_12 = torch.expm1(r2 * h)
                phi_1 = torch.expm1(h)
                phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
                phi_2 = phi_1 / h - 1.0
                phi_3 = phi_2 / h - 0.5
                if model_s is None:
                    model_s = model_fn(x, s)
                if model_s1 is None:
                    x_s1 = (
                        torch.exp(log_alpha_s1 - log_alpha_s) * x
                        - sigma_s1 * phi_11 * model_s
                    )
                    model_s1 = model_fn(x_s1, s1)
                x_s2 = (
                    torch.exp(log_alpha_s2 - log_alpha_s) * x
                    - sigma_s2 * phi_12 * model_s
                    - r2 / r1 * sigma_s2 * phi_22 * (model_s1 - model_s)
                )
                model_s2 = model_fn(x_s2, s2)
                if self.default_high_order_solvers:
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - sigma_t * phi_1 * model_s
                        - (1.0 / r2) * sigma_t * phi_2 * (model_s2 - model_s)
                    )
                else:
                    D1_0 = (1.0 / r1) * (model_s1 - model_s)
                    D1_1 = (1.0 / r2) * (model_s2 - model_s)
                    D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                    D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - sigma_t * phi_1 * model_s
                        - sigma_t * phi_2 * D1
                        - sigma_t * phi_3 * D2
                    )
            if return_intermediate:
                return x_t, {
                    "model_s": model_s,
                    "model_s1": model_s1,
                    "model_s2": model_s2,
                }
            else:
                return x_t

        def multistep_dpm_solver_second_update(
            x: Union[torch.Tensor, TensorDict],
            model_prev_list: List[Union[torch.Tensor, TensorDict]],
            t_prev_list: List[torch.Tensor],
            t: torch.Tensor,
        ):
            """
            Overview:
                Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                model_prev_list (:obj:`List[Union[torch.Tensor, TensorDict]]`): The previous computed model values.
                t_prev_list (:obj:`List[torch.Tensor]`): The previous times, each time has the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            model_prev_1, model_prev_0 = model_prev_list
            t_prev_1, t_prev_0 = t_prev_list
            lambda_prev_1 = diffusion_process.HalfLogSNR(t=t_prev_1, x=x)
            lambda_prev_0 = diffusion_process.HalfLogSNR(t=t_prev_0, x=x)
            lambda_t = diffusion_process.HalfLogSNR(t=t, x=x)
            log_alpha_prev_0 = diffusion_process.log_scale(t=t_prev_0, x=x)
            log_alpha_t = diffusion_process.log_scale(t=t, x=x)
            sigma_prev_0 = diffusion_process.std(t=t_prev_0, x=x)
            sigma_t = diffusion_process.std(t=t, x=x)
            alpha_t = torch.exp(log_alpha_t)

            h_0 = lambda_prev_0 - lambda_prev_1
            h = lambda_t - lambda_prev_0
            r0 = h_0 / h
            D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
            if self.use_dpm_solver_plus_plus:
                if self.solver_type == "dpm_solver":
                    x_t = (
                        sigma_t / sigma_prev_0 * x
                        - alpha_t * (torch.exp(-h) - 1.0) * model_prev_0
                        - 0.5 * alpha_t * (torch.exp(-h) - 1.0) * D1_0
                    )
                elif self.solver_type == "taylor":
                    x_t = (
                        sigma_t / sigma_prev_0 * x
                        - alpha_t * (torch.exp(-h) - 1.0) * model_prev_0
                        + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1_0
                    )
            else:
                if self.solver_type == "dpm_solver":
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_prev_0) * x
                        - sigma_t * (torch.exp(h) - 1.0) * model_prev_0
                        - 0.5 * sigma_t * (torch.exp(h) - 1.0) * D1_0
                    )
                elif self.solver_type == "taylor":
                    x_t = (
                        torch.exp(log_alpha_t - log_alpha_prev_0) * x
                        - sigma_t * (torch.exp(h) - 1.0) * model_prev_0
                        - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1_0
                    )
            return x_t

        def multistep_dpm_solver_third_update(
            x: Union[torch.Tensor, TensorDict],
            model_prev_list: List[Union[torch.Tensor, TensorDict]],
            t_prev_list: List[torch.Tensor],
            t: torch.Tensor,
        ):
            """
            Overview:
                Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                model_prev_list (:obj:`List[Union[torch.Tensor, TensorDict]]`): The previous computed model values.
                t_prev_list (:obj:`List[torch.Tensor]`): The previous times, each time has the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            model_prev_2, model_prev_1, model_prev_0 = model_prev_list
            t_prev_2, t_prev_1, t_prev_0 = t_prev_list
            lambda_prev_2 = diffusion_process.HalfLogSNR(t=t_prev_2, x=x)
            lambda_prev_1 = diffusion_process.HalfLogSNR(t=t_prev_1, x=x)
            lambda_prev_0 = diffusion_process.HalfLogSNR(t=t_prev_0, x=x)
            lambda_t = diffusion_process.HalfLogSNR(t=t, x=x)
            log_alpha_prev_0 = diffusion_process.log_scale(t=t_prev_0, x=x)
            log_alpha_t = diffusion_process.log_scale(t=t, x=x)
            sigma_prev_0 = diffusion_process.std(t=t_prev_0, x=x)
            sigma_t = diffusion_process.std(t=t, x=x)
            alpha_t = torch.exp(log_alpha_t)

            h_1 = lambda_prev_1 - lambda_prev_2
            h_0 = lambda_prev_0 - lambda_prev_1
            h = lambda_t - lambda_prev_0
            r0, r1 = h_0 / h, h_1 / h
            D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
            D1_1 = 1.0 / r1 * (model_prev_1 - model_prev_2)
            D1 = D1_0 + r0 / (r0 + r1) * (D1_0 - D1_1)
            D2 = 1.0 / (r0 + r1) * (D1_0 - D1_1)
            if self.use_dpm_solver_plus_plus:
                x_t = (
                    sigma_t / sigma_prev_0 * x
                    - alpha_t * (torch.exp(-h) - 1.0) * model_prev_0
                    + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1
                    - alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5) * D2
                )
            else:
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_prev_0) * x
                    - sigma_t * (torch.exp(h) - 1.0) * model_prev_0
                    - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1
                    - sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5) * D2
                )
            return x_t

        def singlestep_dpm_solver_update(
            x: Union[torch.Tensor, TensorDict],
            s: torch.Tensor,
            t: torch.Tensor,
            order: float,
            return_intermediate: bool = False,
            r1: float = None,
            r2: float = None,
        ):
            """
            Overview:
                Singlestep DPM-Solver with the order `order` from time `s` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                s (:obj:`torch.Tensor`): The starting time, with the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
                order (:obj:`int`): The order of DPM-Solver. We only support order == 1 or 2 or 3.
                return_intermediate (:obj:`bool`): If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
                r1 (:obj:`float`): The hyperparameter of the second-order or third-order solver.
                r2 (:obj:`float`): The hyperparameter of the third-order solver.
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """
            if order == 1:
                return dpm_solver_first_update(
                    x, s, t, return_intermediate=return_intermediate
                )
            elif order == 2:
                return singlestep_dpm_solver_second_update(
                    x, s, t, return_intermediate=return_intermediate, r1=r1
                )
            elif order == 3:
                return singlestep_dpm_solver_third_update(
                    x, s, t, return_intermediate=return_intermediate, r1=r1, r2=r2
                )
            else:
                raise ValueError(
                    "Solver order must be 1 or 2 or 3, got {}".format(order)
                )

        def multistep_dpm_solver_update(
            x: Union[torch.Tensor, TensorDict],
            model_prev_list: List[Union[torch.Tensor, TensorDict]],
            t_prev_list: List[torch.Tensor],
            t: torch.Tensor,
            order: int,
        ):
            """
            Overview:
                Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `s`.
                model_prev_list (:obj:`List[Union[torch.Tensor, TensorDict]]`): The previous computed model values.
                t_prev_list (:obj:`List[torch.Tensor]`): The previous times, each time has the shape (x.shape[0],).
                t (:obj:`torch.Tensor`): The ending time, with the shape (x.shape[0],).
                order (:obj:`int`): The order of DPM-Solver. We only support order == 1 or 2 or 3.
            Returns:
                x_t (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t`.
            """

            if order == 1:
                return dpm_solver_first_update(
                    x, t_prev_list[-1], t, model_s=model_prev_list[-1]
                )
            elif order == 2:
                return multistep_dpm_solver_second_update(
                    x, model_prev_list, t_prev_list, t
                )
            elif order == 3:
                return multistep_dpm_solver_third_update(
                    x, model_prev_list, t_prev_list, t
                )
            else:
                raise ValueError(
                    "Solver order must be 1 or 2 or 3, got {}".format(order)
                )

        def dpm_solver_adaptive(
            x: Union[torch.Tensor, TensorDict],
            t_T: float,
            t_0: float,
            h_init: float = 0.05,
            theta: float = 0.9,
            t_err: float = 1e-5,
            save_intermediate: bool = False,
        ):
            """
            Overview:
                The adaptive step size solver based on singlestep DPM-Solver.
            Arguments:
                x (:obj:`Union[torch.Tensor, TensorDict]`): The initial value at time `t_T`.
                t_T (:obj:`float`): The starting time of the sampling (default is T).
                t_0 (:obj:`float`): The ending time of the sampling (default is epsilon).
                h_init (:obj:`float`): The initial step size (for logSNR).
                theta (:obj:`float`): The safety hyperparameter for adapting the step size.
                t_err (:obj:`float`): The tolerance for the time.
                save_intermediate (:obj:`bool`): If true, also return the intermediate values.
            Returns:
                x_0 (:obj:`Union[torch.Tensor, TensorDict]`): The approximated solution at time `t_0`.
            References:
                [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, \
                    "Gotta go fast when generating data with score-based models," \
                    arXiv preprint arXiv:2105.14080, 2021.
            """
            s = t_T * torch.ones((x.shape[0],)).to(x)
            lambda_s = diffusion_process.HalfLogSNR(t=t_T, x=x)
            lambda_0 = diffusion_process.HalfLogSNR(t=t_0, x=x)
            h = h_init * torch.ones_like(s).to(x)
            x_prev = x
            x_list = []
            if self.order == 2:
                r1 = 0.5
                lower_update = lambda x, s, t: dpm_solver_first_update(
                    x, s, t, return_intermediate=True
                )
                higher_update = (
                    lambda x, s, t, **kwargs: singlestep_dpm_solver_second_update(
                        x, s, t, r1=r1, **kwargs
                    )
                )
            elif self.order == 3:
                r1, r2 = 1.0 / 3.0, 2.0 / 3.0
                lower_update = lambda x, s, t: singlestep_dpm_solver_second_update(
                    x, s, t, r1=r1, return_intermediate=True
                )
                higher_update = (
                    lambda x, s, t, **kwargs: singlestep_dpm_solver_third_update(
                        x, s, t, r1=r1, r2=r2, **kwargs
                    )
                )
            else:
                raise ValueError(
                    "For adaptive step size solver, order must be 2 or 3, got {}".format(
                        self.order
                    )
                )
            while torch.abs((s - t_0)).mean() > t_err:
                t = diffusion_process.InverseHalfLogSNR(HalfLogSNR=lambda_s + h)[:, 0]
                x_lower, lower_noise_kwargs = lower_update(x, s, t)
                x_higher = higher_update(x, s, t, **lower_noise_kwargs)
                delta = torch.max(
                    torch.ones_like(x).to(x) * self.atol,
                    self.rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)),
                )
                norm_fn = lambda v: torch.sqrt(
                    torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True)
                )
                E = norm_fn((x_higher - x_lower) / delta).max()
                if save_intermediate:
                    x_list.append(x_higher.clone())
                if torch.all(E <= 1.0):
                    x = x_higher
                    s = t
                    x_prev = x_lower
                    lambda_s = diffusion_process.HalfLogSNR(t=t, x=x)
                h = torch.min(
                    theta * h * torch.float_power(E, -1.0 / self.order).float(),
                    lambda_0 - lambda_s,
                )
                self.nfe += self.order
            if save_intermediate:
                return x, x_list
            else:
                return x

        t_0 = 0.00001
        t_T = diffusion_process.t_max
        if save_intermediate:
            x_list = [x.clone()]

        if self.method == "adaptive":
            with torch.no_grad():
                if save_intermediate:
                    x, x_list_ = dpm_solver_adaptive(
                        x, t_T=t_T, t_0=t_0, save_intermediate=True
                    )
                    x_list.extend(x_list_)
                else:
                    x = dpm_solver_adaptive(
                        x, t_T=t_T, t_0=t_0, save_intermediate=False
                    )
        elif self.method == "multistep":
            assert steps >= self.order
            timesteps = get_time_steps(t_T=t_T, t_0=t_0, N=steps)
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for init_order in range(1, self.order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x = multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, vec_t, init_order
                    )
                    if save_intermediate:
                        x_list.append(x.clone())
                    model_prev_list.append(model_fn(x, vec_t))
                    t_prev_list.append(vec_t)
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(self.order, steps + 1):
                    vec_t = timesteps[step].expand(x.shape[0])
                    x = multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, vec_t, self.order
                    )
                    if save_intermediate:
                        x_list.append(x.clone())
                    for i in range(self.order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = model_fn(x, vec_t)
        elif self.method in ["singlestep", "singlestep_fixed"]:
            if self.method == "singlestep":
                orders = get_orders_for_singlestep_solver(steps=steps, order=self.order)
                timesteps = get_time_steps(t_T=t_T, t_0=t_0, N=steps)
            elif self.method == "singlestep_fixed":
                K = steps // order
                orders = [
                    order,
                ] * K
                timesteps = get_time_steps(t_T=t_T, t_0=t_0, N=(K * order))
            with torch.no_grad():
                i = 0
                for order in orders:
                    vec_s, vec_t = timesteps[i].expand(x.shape[0]), timesteps[
                        i + order
                    ].expand(x.shape[0])
                    h = diffusion_process.HalfLogSNR(
                        t=timesteps[i + order]
                    ) - diffusion_process.HalfLogSNR(t=timesteps[i])
                    r1 = (
                        None
                        if order <= 1
                        else (
                            diffusion_process.HalfLogSNR(t=timesteps[i + 1])
                            - diffusion_process.HalfLogSNR(t=timesteps[i])
                        )
                        / h
                    )
                    r2 = (
                        None
                        if order <= 2
                        else (
                            diffusion_process.HalfLogSNR(t=timesteps[i + 2])
                            - diffusion_process.HalfLogSNR(t=timesteps[i])
                        )
                        / h
                    )
                    x = singlestep_dpm_solver_update(
                        x, vec_s, vec_t, order, r1=r1, r2=r2
                    )
                    if save_intermediate:
                        x_list.append(x.clone())
                    i += order
        if self.denoise:
            x = denoise_fn(x, torch.ones((x.shape[0],)).to(self.device) * t_0)
            if save_intermediate:
                x_list[-1] = x.clone()
        if save_intermediate:
            return torch.stack(x_list, dim=0)
        else:
            return x
