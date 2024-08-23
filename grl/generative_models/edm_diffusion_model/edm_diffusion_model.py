from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import treetensor
from tensordict import TensorDict

import torch.optim as optim
from easydict import EasyDict
from functools import partial

from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.random_generator import gaussian_random_variable
from grl.numerical_methods.numerical_solvers import get_solver
from grl.numerical_methods.numerical_solvers.dpm_solver import DPMSolver
from grl.numerical_methods.numerical_solvers.ode_solver import (
    DictTensorODESolver,
    ODESolver,
)
from grl.numerical_methods.numerical_solvers.sde_solver import SDESolver

from grl.utils import find_parameters
from grl.utils import set_seed
from grl.utils.log import log

from .edm_preconditioner import PreConditioner
from .edm_utils import SIGMA_T, SIGMA_T_DERIV, SIGMA_T_INV
from .edm_utils import SCALE_T, SCALE_T_DERIV
from .edm_utils import INITIAL_SIGMA_MAX, INITIAL_SIGMA_MIN
from .edm_utils import DEFAULT_PARAM, DEFAULT_SOLVER_PARAM


class EDMModel(nn.Module):
    """
    Overview:
        An implementation of EDM, which eludicates diffusion based generative model through preconditioning, training, sampling.
        This implementation supports 4 types: `VP_edm`(DDPM-SDE), `VE_edm` (SGM-SDE), `iDDPM_edm`, `EDM`. More details see Table 1 in paper
        EDM class utilizes different params and executes different scheules during precondition, training and sample process. 
        Sampling supports 1st order Euler step and 2nd order Heun step as Algorithm 1 in paper.
        For EDM type itself, stochastic sampler as Algorithm 2 in paper is also supported.    
    Interface:
        ``__init__``, ``forward``, ``sample``
    Reference:
        EDM original paper: https://arxiv.org/abs/2206.00364
        Code reference: https://github.com/NVlabs/edm
    """
    def __init__(self, config: Optional[EasyDict]=None) -> None:
        
        super().__init__()
        self.config = config
        self.x_size = config.x_size
        self.device = config.device
        

        self.gaussian_generator = gaussian_random_variable(
            config.x_size,
            config.device,
            config.use_tree_tensor if hasattr(config, "use_tree_tensor") else False,
        )

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

        # EDM Type ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"]
        self.edm_type = config.path.edm_type
        assert self.edm_type in ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"], \
            f"Your edm type should in 'VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM'], but got {self.edm_type}"
        
        #* 1. Construct basic Unet architecture through params in config
        self.model = IntrinsicModel(config.model.args)

        #* 2. Precond setup
        self.params = EasyDict(DEFAULT_PARAM[self.edm_type])
        self.params.update(config.path.params)
        log.info(f"Using edm type: {self.edm_type}\nParam is {self.params}")
        self.preconditioner = PreConditioner(
            self.edm_type, 
            denoise_model=self.model, 
            use_mixes_precision=False,
            **self.params
        )
       
        self.solver_params = EasyDict(DEFAULT_SOLVER_PARAM)
        self.solver_params.update(config.sample_params)

        # Initialize sigma_min and sigma_max if not provided
        
        self.sigma_min = INITIAL_SIGMA_MIN[self.edm_type] if "sigma_min" not in self.params else self.params.sigma_min          
        self.sigma_max = INITIAL_SIGMA_MAX[self.edm_type] if "sigma_max" not in self.params else self.params.sigma_max          

    @property       
    def get_type(self) -> str:
        return "EDMModel"

    def _sample_sigma_weight_train(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Overview:
            Sample sigma from given distribution for training according to edm type.
            
        Arguments:
            x (:obj:`torch.Tensor`): The sample which needs to add noise.
            
        Returns:
            sigma (:obj:`torch.Tensor`): Sampled sigma from the distribution.
            weight (:obj:`torch.Tensor`): Loss weight obtained from sampled sigma.
        """
        # assert the first dim of x is batch size

        rand_shape = [x.shape[0]] + [1] * (x.ndim - 1) 
        if self.edm_type == "VP_edm":
            epsilon_t = self.params.epsilon_t
            beta_d = self.params.beta_d
            beta_min = self.params.beta_min
            
            rand_uniform = torch.rand(*rand_shape, device=self.device)
            sigma = SIGMA_T["VP_edm"](1 + rand_uniform * (epsilon_t - 1), beta_d, beta_min)
            weight = 1 / sigma ** 2
        elif self.edm_type == "VE_edm":
            rand_uniform = torch.rand(*rand_shape, device=self.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rand_uniform)
            weight = 1 / sigma ** 2
        elif self.edm_type == "iDDPM_edm":
            u = self.preconditioner.u
            sigma_index = torch.randint(0, self.params.M - 1, rand_shape, device=self.device)
            sigma = u[sigma_index]
            weight = 1 / sigma ** 2
        elif self.edm_type == "EDM":
            P_mean = self.params.P_mean
            P_std = self.params.P_std
            sigma_data = self.params.sigma_data
            
            rand_normal = torch.randn(*rand_shape, device=self.device)
            sigma = (rand_normal * P_std + P_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        return sigma, weight
    
    def forward(self, x, condition=None):
        return self.sample(x, condition)
    
    def L2_denoising_matching_loss(
            self,
            x: Tensor,
            condition: Optional[Tensor]=None
        ): 
        """
        Overview:
            Calculate the L2 denoising matching loss.
        Arguments:
            x (:obj:`torch.Tensor`): The sample which needs to add noise.
            condition (:obj:`torch.Tensor`): The condition for the sample. Default setting: None.
        Returns:
            loss (:obj:`torch.Tensor`): The L2 denoising matching loss.
        """

        sigma, weight = self._sample_sigma_weight_train(x)
        n = torch.randn_like(x) * sigma
        D_xn = self.preconditioner(sigma, x+n, condition=condition)
        loss = weight * ((D_xn - x) ** 2)
        return loss.mean()

    def _get_sigma_steps_t_steps(self, 
                                 num_steps: int=18, 
                                 epsilon_s: float=1e-3, rho: Union[int, float]=7
                            ):
        """
        Overview:
            Get the schedule of sigma according to differernt t schedules.
            
        Arguments:
            num_steps (:obj:`int`): The number of timesteps during denoise sampling. Default setting: 18.
            epsilon_s (:obj:`float`): Parameter epsilon_s (only VP_edm needs).
            rho (:obj:`Union[int, float]`): Parameter rho (only EDM needs).  
        
        Returns:
            sigma_steps (:obj:`torch.Tensor`): The scheduled sigma.
            t_steps (:obj:`torch.Tensor`): The scheduled t.
        
        """
        # Define time steps in terms of noise level
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        sigma_steps = None
        if self.edm_type == "VP_edm":
            vp_beta_d = 2 * (np.log(self.sigma_min ** 2 + 1) / epsilon_s - np.log(self.sigma_max ** 2 + 1)) / (epsilon_s - 1)
            vp_beta_min = np.log(self.sigma_max ** 2 + 1) - 0.5 * vp_beta_d
            
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = SIGMA_T[self.edm_type](orig_t_steps, vp_beta_d, vp_beta_min)
        
        elif self.edm_type == "VE_edm":
            orig_t_steps = (self.sigma_max ** 2) * ((self.sigma_min ** 2 / self.sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = SIGMA_T[self.edm_type](orig_t_steps)
        
        elif self.edm_type == "iDDPM_edm":
            M, C_1, C_2 = self.params.M, self.params.C_1, self.params.C_2
            
            u = torch.zeros(M + 1, dtype=torch.float, device=self.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(self.params.M, 0, -1, device=self.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)]
            
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]   
            orig_t_steps = SIGMA_T_INV[self.edm_type](self.preconditioner.round_sigma(sigma_steps))         
        
        elif self.edm_type == "EDM": 
            sigma_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * \
                (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
            orig_t_steps = SIGMA_T_INV[self.edm_type](self.preconditioner.round_sigma(sigma_steps))         
        else:
            raise NotImplementedError(f"Please check your edm_type: {self.edm_type}, which is not in ['VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM']")
        
        t_steps = torch.cat([orig_t_steps, torch.zeros_like(orig_t_steps[:1])]) # t_N = 0E
        
        return sigma_steps, t_steps  
    
    
    def _get_sigma_deriv_inv_scale_deriv(self, epsilon_s: Union[int, float]=1e-3) \
                                -> Tuple[Callable, Callable, Callable, Callable, Callable]:
        """
        Overview:
            Get sigma(t) for different solver schedules.
            
        Returns:
            sigma: (:obj:`Callable`): sigma(t)
            sigma_deriv: (:obj:`Callable`): sigma'(t)
            sigma_inv: (:obj:`Callable`): sigma^{-1} (sigma) 
            scale: (:obj:`Callable`): s(t)
            scale_deriv: (:obj:`Callable`): s'(t)
            
        """
        vp_beta_d = 2 * (np.log(self.sigma_min ** 2 + 1) / epsilon_s - np.log(self.sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(self.sigma_max ** 2 + 1) - 0.5 * vp_beta_d
        sigma = partial(SIGMA_T[self.edm_type], beta_d=vp_beta_d, beta_min=vp_beta_min)
        sigma_deriv = partial(SIGMA_T_DERIV[self.edm_type], beta_d=vp_beta_d, beta_min=vp_beta_min)
        sigma_inv = partial(SIGMA_T_INV[self.edm_type], beta_d=vp_beta_d, beta_min=vp_beta_min)
        scale = partial(SCALE_T[self.edm_type], beta_d=vp_beta_d, beta_min=vp_beta_min)
        scale_deriv = partial(SCALE_T_DERIV[self.edm_type], beta_d=vp_beta_d, beta_min=vp_beta_min)

        return sigma, sigma_deriv, sigma_inv, scale, scale_deriv

    def sample(
            self, 
            t_span: torch.Tensor = None,
            batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
            x_0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
            condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
            with_grad: bool = False,
            solver_config: EasyDict = None,
        ):

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
            condition = torch.repeat_interleave(
                condition, torch.prod(extra_batch_size), dim=0
            )
            # condition.shape = (B*N, D)

        
        sigma_steps, t_steps = self._get_sigma_steps_t_steps(num_steps=self.solver_params.num_steps, epsilon_s=self.solver_params.epsilon_s, rho=self.solver_params.rho)

        sigma, sigma_deriv, sigma_inv, scale, scale_deriv = self._get_sigma_deriv_inv_scale_deriv()
                
        S_churn = self.solver_params.S_churn
        S_min = self.solver_params.S_min
        S_max = self.solver_params.S_max
        S_noise = self.solver_params.S_noise
        alpha = self.solver_params.alpha
        
        # # Main sampling loop
        # t_next = t_steps[0]
        # x_next = x_0 * (sigma(t_next) * scale(t_next))
        # for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        #     x_cur = x_next

        #     # Euler step.
        #     h = t_next - t_cur
        #     denoised = self.preconditioner(sigma(t_cur), x_cur / scale(t_cur), condition)
        #     d_cur = (sigma_deriv(t_cur) / sigma(t_cur) + scale_deriv(t_cur) / scale(t_cur)) * x_cur - sigma_deriv(t_cur) * scale(t_cur) / sigma(t_cur) * denoised
            
        #     x_next = x_cur + h * d_cur

        def drift(t, x):
            t_shape = [x.shape[0]] + [1] * (x.ndim - 1)
            t = t.view(*t_shape)
            denoised = self.preconditioner(sigma(t), x / scale(t), condition)
            f=(sigma_deriv(t) / sigma(t) + scale_deriv(t) / scale(t)) * x - sigma_deriv(t) * scale(t) / sigma(t) * denoised
            return f

        t_span = torch.tensor(t_steps, device=self.device)
        if isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
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