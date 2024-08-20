from typing import Optional, Tuple, Literal
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict

from .edm_preconditioner import PreConditioner
from grl.generative_models.intrinsic_model import IntrinsicModel

class EDMModel(nn.Module):
    
    def __init__(self, config: Optional[EasyDict]=None) -> None:
        
        super().__init__()
        self.config: EasyDict = config
        self.device: torch.device = config.device
        
        # EDM Type ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"]
        self.edm_type: str = config.edm_model.path.edm_type
        assert self.edm_type in ["VP_edm", "VE_edm", "iDDPM_edm", "EDM"], \
            f"Your edm type should in 'VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM'], but got {self.edm_type}"
        
        #* 1. Construct basic Unet architecture through params in config
        # TODO: construct basic denoise network here
        
        self.base_denoise_network: Optional[nn.Module] = IntrinsicModel(config.edm_model.model.args)

        #* 2. Precond setup
        self.params: EasyDict = config.edm_model.path.params
        self.preconditioner = PreConditioner(
            self.edm_type, 
            base_denoise_model=self.base_denoise_network, 
            use_mixes_precision=False,
            **self.params
        )
        
        #* 3. Solver setup
        self.solver_type: str = config.edm_model.solver.solver_type
        self.solver_schedule: str = config.edm_model.solver.schedule
        self.solver_scaling: str = config.edm_model.solver.scaling
        self.solver_params: EasyDict = config.edm_model.solver.params
        assert self.solver_type in ['euler', 'heun']
        assert self.solver_schedule in ['VP', 'VE', 'Linear']
        assert self.solver_scaling in ["VP", "none"]
        
        # Initialize sigma_min and sigma_max if not provided
        
        if "sigma_min" not in self.params:
            self._initialize_sigma_min()
        else:
            self.sigma_min = self.params.sigma_min
        if "sigma_max" not in self.params:
            self._initialize_sigma_max()
        else:
            self.sigma_max = self.params.sigma_max
            
    def get_type(self):
        return "DiffusionModel"
    
    def _initialize_sigma_min(self):
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.exp(0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        self.sigma_min = {
            "VP_edm": vp_sigma(19.9, 0.1)(1e-3), 
            "VE_edm": 0.02, 
            "iDDPM_edm": 0.002, 
            "EDM": 0.002
        }[self.edm_type]

    def _initialize_sigma_max(self):
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.exp(0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        self.sigma_max = {
            "VP_edm": vp_sigma(19.9, 0.1)(1), 
            "VE_edm": 100, 
            "iDDPM_edm": 81, 
            "EDM": 80
        }[self.edm_type]
    
    # For VP_edm
    
    
    def _sample_sigma_weight_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert the first dim of x is batch size
        rand_shape = [x.shape[0]] + [1] * (x.ndim - 1) 
        if self.edm_type == "VP_edm":
            def sigma_for_vp_edm(self, t):
                t = torch.as_tensor(t)
                return ((0.5 * self.params.beta_d * (t ** 2) + self.params.beta_min * t).exp() - 1).sqrt()
            rand_uniform = torch.rand(*rand_shape, device=x.device)
            sigma = sigma_for_vp_edm(1 + rand_uniform * (self.params.epsilon_t - 1))
            weight = 1 / sigma ** 2
        elif self.edm_type == "VE_edm":
            rand_uniform = torch.rand(*rand_shape, device=x.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rand_uniform)
            weight = 1 / sigma ** 2
        elif self.edm_type == "EDM":
            rand_normal = torch.randn(*rand_shape, device=x.device)
            sigma = (rand_normal * self.params.P_std + self.params.P_mean).exp()
            weight = (sigma ** 2 + self.params.sigma_data ** 2) / (sigma * self.params.sigma_data) ** 2
        return sigma, weight
    
    def forward(self, x: torch.Tensor, class_labels=None):
        x = x.to(self.device)
        sigma, weight = self._sample_sigma_weight_train(x)
        n = torch.randn_like(x) * sigma
        D_xn = self.preconditioner(x+n, sigma, class_labels=class_labels)
        loss = weight * ((D_xn - x) ** 2)
        return loss
    
    
    def _get_sigma_steps(self):
        """
        Overview:
            Get the schedule of sigma according to differernt t schedules.
            
        """
        self.sigma_min = max(self.sigma_min, self.preconditioner.sigma_min)
        self.sigma_max = min(self.sigma_max, self.preconditioner.sigma_max)
        
        # Define time steps in terms of noise level
        step_indices = torch.arange(self.solver_params.num_steps, dtype=torch.float64, device=self.device)
        sigma_steps = None
        if self.edm_type == "VP_edm":
            vp_beta_d = 2 * (np.log(self.sigma_min ** 2 + 1) / self.params.epsilon_s - np.log(self.sigma_max ** 2 + 1)) / (self.params.epsilon_s - 1)
            vp_beta_min = np.log(self.sigma_max ** 2 + 1) - 0.5 * vp_beta_d
            vp_sigma = lambda beta_d, beta_min: lambda t: (np.exp((0.5 * beta_d * (t ** 2) + beta_min * t)) - 1) ** 0.5
            orig_t_steps = 1 + step_indices / (self.solver_params.num_steps - 1) * (self.params.epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        
        elif self.edm_type == "VE_edm":
            ve_sigma = lambda t: t.sqrt()
            orig_t_steps = (self.sigma_max ** 2) * ((self.sigma_min ** 2 / self.sigma_max ** 2) ** (step_indices / (self.solver_params.num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        
        elif self.edm_type == "iDDPM_edm":
            u = torch.zeros(self.params.M + 1, dtype=torch.float64, device=self.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / self.params.M / (self.params.C_2 + 1)).sin() ** 2
            for j in torch.arange(self.params.M, 0, -1, device=self.device): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.params.C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (self.solver_params.num_steps - 1) * step_indices).round().to(torch.int64)]            
        
        elif self.edm_type == "EDM": 
            sigma_steps = (self.sigma_max ** (1 / self.solver_params.rho) + step_indices / (self.solver_params.num_steps - 1) * \
                (self.sigma_min ** (1 / self.solver_params.rho) - self.sigma_max ** (1 / self.solver_params.rho))) ** self.solver_params.rho
            # Define noise level schedule.
        return sigma_steps     
    
    
    def _get_sigma_deriv_inv(self):
        """
        Overview:
            Get sigma(t) for different solver schedules.
            
        Returns:
            sigma(t), sigma'(t), sigma^{-1}(sigma) 
        """
        if self.solver_schedule == 'VP': # [VP_edm]
            vp_beta_d = 2 * (np.log(self.sigma_min ** 2 + 1) / self.params.epsilon_s - np.log(self.sigma_max ** 2 + 1)) / (self.params.epsilon_s - 1)
            vp_beta_min = np.log(self.sigma_max ** 2 + 1) - 0.5 * vp_beta_d
            vp_sigma = lambda beta_d, beta_min: lambda t: (np.exp((0.5 * beta_d * (t ** 2) + beta_min * t)) - 1) ** 0.5
            vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
            vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
            vp_beta_d = 2 * (np.log(self.sigma_min ** 2 + 1) / self.params.epsilon_s - np.log(self.sigma_max ** 2 + 1)) / (self.params.epsilon_s - 1)
            vp_beta_min = np.log(self.sigma_max ** 2 + 1) - 0.5 * vp_beta_d
            
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif self.solver_schedule == 'VE':   # [VE_edm]
            sigma = lambda t: t.sqrt()
            sigma_deriv = lambda t: 0.5 / t.sqrt()
            sigma_inv = lambda sigma: sigma ** 2
        elif self.solver_schedule == 'Linear': # [iDDPM_edm, EDM]
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma
            
        return sigma, sigma_deriv, sigma_inv
    
    
    def _get_scaling(self, sigma, sigma_deriv, sigma_inv, sigma_steps):
        """
        Overview:
            Get s(t) for different solver schedules. and t_steps
            
        Returns:
            sigma(t), sigma'(t), sigma^{-1}(sigma) 
        """
        # Define scaling schedule.
        if self.solver_scaling == 'VP':
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        elif self.solver_scaling == 'none': # [VE_edm, iDDPM_edm, EDM]
            s = lambda t: 1
            s_deriv = lambda t: 0
        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(self.preconditioner.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        return s, s_deriv, t_steps
   
    
    def sample(self, latents, class_labels=None, use_stochastic=False):
        # Get sigmas, scales, and timesteps
        latents = latents.to(self.device)
        sigma_steps = self._get_sigma_steps()
        sigma, sigma_deriv, sigma_inv = self._get_sigma_deriv_inv()
        s, s_deriv, t_steps = self._get_scaling(sigma, sigma_deriv, sigma_inv, sigma_steps)
        
        if not use_stochastic:
            # Main sampling loop
            t_next = t_steps[0]
            x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = min(self.solver_params.S_churn / self.solver_params.num_steps, np.sqrt(2) - 1) if self.solver_params.S_min <= sigma(t_cur) <= self.solver_params.S_max else 0
                t_hat = sigma_inv(self.preconditioner.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
                x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * self.solver_params.S_noise * torch.randn_like(x_cur)

                # Euler step.
                h = t_next - t_hat
                denoised = self.preconditioner(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
                d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
                x_prime = x_hat + self.solver_params.alpha * h * d_cur
                t_prime = t_hat + self.solver_params.alpha * h

                # Apply 2nd order correction.
                if self.solver_type == 'euler' or i == self.solver_params.num_steps - 1:
                    x_next = x_hat + h * d_cur
                else:
                    assert self.solver_type == 'heun'
                    denoised = self.preconditioner(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
                    d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
                    x_next = x_hat + h * ((1 - 1 / (2 * self.solver_params.alpha)) * d_cur + 1 / (2 * self.solver_params.alpha) * d_prime)
        
        else:
            assert self.edm_type == "EDM", f"Stochastic can only use in EDM, but your precond type is {self.edm_type}"
            x_next = latents.to(torch.float64) * t_steps[0]
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                x_cur = x_next

                # Increase noise temporarily.
                gamma = min(self.solver_params.S_churn / self.solver_params.num_steps, np.sqrt(2) - 1) if self.solver_params.S_min <= t_cur <= self.solver_params.S_max else 0
                t_hat = self.preconditioner.round_sigma(t_cur + gamma * t_cur)
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.solver_params.S_noise * torch.randn_like(x_cur)

                # Euler step.
                denoised = self.preconditioner(x_hat, t_hat, class_labels).to(torch.float64)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                # Apply 2nd order correction.
                if i < self.solver_params.num_steps - 1:
                    denoised = self.preconditioner(x_next, t_next, class_labels).to(torch.float64)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


        return x_next
