from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict
from functools import partial

from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.utils import find_parameters
from grl.utils import set_seed
from grl.utils.log import log

from .edm_preconditioner import PreConditioner
from .edm_utils import SIGMA_T, SIGMA_T_DERIV, SIGMA_T_INV
from .edm_utils import SCALE_T, SCALE_T_DERIV
from .edm_utils import INITIAL_SIGMA_MAX, INITIAL_SIGMA_MIN
from .edm_utils import DEFAULT_PARAM, DEFAULT_SOLVER_PARAM

class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32), 
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x, noise, class_labels=None):
        return self.model(x)

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
        # self.x_size = config.x_size
        self.device = config.device
        
        # EDM Type ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"]
        self.edm_type = config.edm_model.path.edm_type
        assert self.edm_type in ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"], \
            f"Your edm type should in 'VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM'], but got {self.edm_type}"
        
        #* 1. Construct basic Unet architecture through params in config
        self.base_denoise_network = Simple()

        #* 2. Precond setup
        self.params = EasyDict(DEFAULT_PARAM[self.edm_type])
        self.params.update(config.edm_model.path.params)
        log.info(f"Using edm type: {self.edm_type}\nParam is {self.params}")
        self.preconditioner = PreConditioner(
            self.edm_type, 
            base_denoise_model=self.base_denoise_network, 
            use_mixes_precision=False,
            **self.params
        )
        
        #* 3. Solver setup
        self.solver_type = config.edm_model.solver.solver_type
        assert self.solver_type in ['euler', 'heun'], \
            f"Your solver type should in ['euler', 'heun'], but got {self.solver_type}"
        
        self.solver_params = EasyDict(DEFAULT_SOLVER_PARAM)
        self.solver_params.update(config.edm_model.solver.params)
        log.info(f"Using solver type: {self.solver_type}\nSolver param is {self.solver_params}")
        # Initialize sigma_min and sigma_max if not provided
        
        
        self.sigma_min = INITIAL_SIGMA_MIN[self.edm_type] if "sigma_min" not in self.params else self.params.sigma_min          
        self.sigma_max = INITIAL_SIGMA_MAX[self.edm_type] if "sigma_max" not in self.params else self.params.sigma_max          

    @property       
    def get_type(self) -> str:
        return "EDMModel"

    def _sample_sigma_weight_train(self, x: Tensor, **params) -> Tuple[Tensor, Tensor]:
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
        params = EasyDict(params)
        rand_shape = [x.shape[0]] + [1] * (x.ndim - 1) 
        if self.edm_type == "VP_edm":
            epsilon_t = params.epsilon_t
            beta_d = params.beta_d
            beta_min = params.beta_min
            
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
            P_mean = params.P_mean
            P_std = params.P_std
            sigma_data = params.sigma_data
            
            rand_normal = torch.randn(*rand_shape, device=self.device)
            sigma = (rand_normal * P_std + P_mean).exp()
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        return sigma, weight
    
    def forward(self, x: Tensor, class_labels: Tensor=None) -> Tensor:
        x = x.to(self.device)
        sigma, weight = self._sample_sigma_weight_train(x, **self.params)
        n = torch.randn_like(x) * sigma
        D_xn = self.preconditioner(x+n, sigma, class_labels=class_labels)
        loss = weight * ((D_xn - x) ** 2)
        return loss.mean()
    
    
    def _get_sigma_steps_t_steps(self, 
                                 num_steps: int=18, 
                                 epsilon_s: float=1e-3, rho: Union[int, float]=7
                            )-> Tuple[Tensor, Tensor]:
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
        self.sigma_min = max(self.sigma_min, self.preconditioner.sigma_min)
        self.sigma_max = min(self.sigma_max, self.preconditioner.sigma_max)
    
        # Define time steps in terms of noise level
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
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
            
            u = torch.zeros(M + 1, dtype=torch.float64, device=self.device)
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
  
    
    def sample(self, 
               t_span, 
               batch_size,
               latents: Tensor, 
               class_labels: Tensor=None, 
               use_stochastic: bool=False, 
               **solver_kwargs
        ) -> Tensor:
        
        # Get sigmas, scales, and timesteps
        log.info(f"Start sampling!")
        num_steps = self.solver_params.num_steps
        epsilon_s = self.solver_params.epsilon_s
        rho = self.solver_params.rho
        
        latents = latents.to(self.device)
        sigma_steps, t_steps = self._get_sigma_steps_t_steps(num_steps=num_steps, epsilon_s=epsilon_s, rho=rho)
        sigma, sigma_deriv, sigma_inv, scale, scale_deriv = self._get_sigma_deriv_inv_scale_deriv()
                
        S_churn = self.solver_params.S_churn
        S_min = self.solver_params.S_min
        S_max = self.solver_params.S_max
        S_noise = self.solver_params.S_noise
        alpha = self.solver_params.alpha
        
        
        if not use_stochastic:
            # Main sampling loop
            t_next = t_steps[0]
            x_next = latents * (sigma(t_next) * scale(t_next))
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
                t_hat = sigma_inv(self.preconditioner.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
                x_hat = scale(t_hat) / scale(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * scale(t_hat) * S_noise * torch.randn_like(x_cur)

                # Euler step.
                h = t_next - t_hat
                denoised = self.preconditioner(x_hat / scale(t_hat), sigma(t_hat), class_labels)
                d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + scale_deriv(t_hat) / scale(t_hat)) * x_hat - sigma_deriv(t_hat) * scale(t_hat) / sigma(t_hat) * denoised
                x_prime = x_hat + alpha * h * d_cur
                t_prime = t_hat + alpha * h

                # Apply 2nd order correction.
                if self.solver_type == 'euler' or i == num_steps - 1:
                    x_next = x_hat + h * d_cur
                else:
                    assert self.solver_type == 'heun'
                    denoised = self.preconditioner(x_prime / scale(t_prime), sigma(t_prime), class_labels)
                    d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + scale_deriv(t_prime) / scale(t_prime)) * x_prime - sigma_deriv(t_prime) * scale(t_prime) / sigma(t_prime) * denoised
                    x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        
        else:
            assert self.edm_type == "EDM", f"Stochastic can only use in EDM, but your precond type is {self.edm_type}"
            x_next = latents * t_steps[0]
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                t_hat = self.preconditioner.round_sigma(t_cur + gamma * t_cur)
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

                # Euler step.
                denoised = self.preconditioner(x_hat, t_hat, class_labels)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < num_steps - 1:
                    denoised = self.preconditioner(x_next, t_next, class_labels)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


        return x_next
