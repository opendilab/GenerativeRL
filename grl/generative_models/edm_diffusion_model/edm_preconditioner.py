from typing import Optional, Tuple, Literal
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, as_tensor
import torch.nn as nn
import torch.nn.functional as F

from .edm_utils import SIGMA_T, SIGMA_T_INV

class PreConditioner(nn.Module):
    
    def __init__(self, 
                 precondition_type: Literal["VP_edm", "VE_edm", "iDDPM_edm", "EDM"] = "EDM",
                 base_denoise_model: nn.Module = None, 
                 use_mixes_precision: bool = False, 
                 **precond_config_kwargs) -> None:
        
        super().__init__()
        self.precondition_type = precondition_type
        self.base_denoise_model = base_denoise_model
        self.use_mixes_precision = use_mixes_precision
        
        if self.precondition_type == "VP_edm":
            self.beta_d = precond_config_kwargs.get("beta_d", 19.9)
            self.beta_min = precond_config_kwargs.get("beta_min", 0.1)
            self.M = precond_config_kwargs.get("M", 1000)
            self.epsilon_t = precond_config_kwargs.get("epsilon_t", 1e-5)
            
            self.sigma_min = SIGMA_T["VP_edm"](self.epsilon_t, self.beta_d, self.beta_min)
            self.sigma_max = SIGMA_T["VP_edm"](1, self.beta_d, self.beta_min)
            
        elif self.precondition_type == "VE_edm":
            self.sigma_min = precond_config_kwargs.get("sigma_min", 0.02)
            self.sigma_max = precond_config_kwargs.get("sigma_max", 100)
            
        elif self.precondition_type == "iDDPM_edm":
            self.C_1 = precond_config_kwargs.get("C_1", 0.001)
            self.C_2 = precond_config_kwargs.get("C_2", 0.008)
            self.M = precond_config_kwargs.get("M", 1000)
            u = torch.zeros(self.M + 1)
            for j in range(self.M, 0, -1): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
            self.register_buffer('u', u)
            self.sigma_min = float(u[self.M - 1])
            self.sigma_max = float(u[0])
            
        elif self.precondition_type == "EDM":
            self.sigma_min = precond_config_kwargs.get("sigma_min", 0.002)
            self.sigma_max = precond_config_kwargs.get("sigma_max", 80)
            self.sigma_data = precond_config_kwargs.get("sigma_data", 0.5)
        
        else:
            raise ValueError(f"Please check your precond type {self.precondition_type} is in ['VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM']")
            

    # For iDDPM_edm
    def alpha_bar(self, j):
        assert self.precondition_type == "iDDPM_edm", f"Only iDDPM_edm supports the alpha bar function, but your precond type is {self.precondition_type}"
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
            

    def round_sigma(self, sigma, return_index=False):
        
        if self.precondition_type == "iDDPM_edm":
            sigma = torch.as_tensor(sigma)
            index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
            result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
            return result.reshape(sigma.shape).to(sigma.device)
        else:
            return torch.as_tensor(sigma)
        
    def get_precondition_c(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if self.precondition_type == "VP_edm":
            c_skip = 1
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_noise = (self.M - 1) * SIGMA_T_INV["VP_edm"](sigma, self.beta_d, self.beta_min)
        elif self.precondition_type == "VE_edm":
            c_skip = 1
            c_out = sigma
            c_in = 1
            c_noise = (0.5 * sigma).log()
        elif self.precondition_type == "iDDPM_edm":
            c_skip = 1
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        elif self.precondition_type == "EDM":
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
            c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise
    
    def forward(self, x: Tensor, sigma: Tensor, class_labels=None, **model_kwargs):
        # Suppose the first dim of x is batch size
        x = x.to(torch.float32)
        sigma_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        if sigma.numel() == 1:
            sigma = sigma.view(-1).expand(*sigma_shape)
        
        dtype = torch.float16 if (self.use_mixes_precision and x.device.type == 'cuda') else torch.float32
        c_skip, c_out, c_in, c_noise = self.get_precondition_c(sigma)
        F_x = self.base_denoise_model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x