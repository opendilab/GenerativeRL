from typing import Union, Optional, Tuple, Literal
from dataclasses import dataclass

from torch import Tensor, as_tensor
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grl.utils.log import log
from grl.generative_models.edm_diffusion_model.edm_utils import SIGMA_T, SIGMA_T_INV

class PreConditioner(nn.Module):
    """
    Overview:
        Precondition step in EDM.
        
    Interface:
        ``__init__``, ``round_sigma``, ``get_precondition_c``, ``forward``
    """
    def __init__(self, 
                 precondition_type: Literal["VP_edm", "VE_edm", "iDDPM_edm", "EDM"] = "EDM",
                 denoise_model: Optional[nn.Module] = None, 
                 use_mixes_precision: bool = False, 
                 **precond_params) -> None:
        """
        Overview:
            Initialize preconditioner for Network preconditioning in EDM.
            More details in Network and Preconditioning in Section 5 of EDM paper.
            
        Arguments:
            precondition_type (:obj:`Literal["VP_edm", "VE_edm", "iDDPM_edm", "EDM"]`): The precond type.
            denoise_model (:obj:`Optional[nn.Module]`): The basic denoise network.
            use_mixes_precision (:obj:`bool`): If mixes precision is used.
        
        Reference:
            EDM original paper link: https://arxiv.org/abs/2206.00364
            Code reference: https://github.com/NVlabs/edm
        """
        super().__init__()
        log.info(f"Precond_params: {precond_params}")
        precond_params = EasyDict(precond_params)
        self.precondition_type = precondition_type
        self.denoise_model = denoise_model
        self.use_mixes_precision = use_mixes_precision
        
        if self.precondition_type == "VP_edm":
            self.beta_d = precond_params.beta_d
            self.beta_min = precond_params.beta_min
            self.M = precond_params.M
            self.epsilon_t = precond_params.epsilon_t
            
            self.sigma_min = float(SIGMA_T["VP_edm"](torch.tensor(self.epsilon_t), self.beta_d, self.beta_min))
            self.sigma_max = float(SIGMA_T["VP_edm"](torch.tensor(1), self.beta_d, self.beta_min))
            
        elif self.precondition_type == "VE_edm":
            self.sigma_min = precond_params.sigma_min
            self.sigma_max = precond_params.sigma_max
            
        elif self.precondition_type == "iDDPM_edm":
            self.C_1 = precond_params.C_1
            self.C_2 = precond_params.C_2
            self.M = precond_params.M
            
            # For iDDPM_edm
            def alpha_bar(j):
                assert self.precondition_type == "iDDPM_edm", f"Only iDDPM_edm supports the alpha bar function, but your precond type is {self.precondition_type}"
                j = torch.as_tensor(j)
                return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
            
            u = torch.zeros(self.M + 1)
            for j in range(self.M, 0, -1): # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
            self.register_buffer('u', u)
            self.sigma_min = float(u[self.M - 1])
            self.sigma_max = float(u[0])
            
        elif self.precondition_type == "EDM":
            self.sigma_min = precond_params.sigma_min
            self.sigma_max = precond_params.sigma_max
            self.sigma_data = precond_params.sigma_data
        
        else:
            raise ValueError(f"Please check your precond type {self.precondition_type} is in ['VP_edm', 'VE_edm', 'iDDPM_edm', 'EDM']")
                    

    def round_sigma(self, sigma: Union[Tensor, float], return_index: bool=False) -> Tensor:
        """
        Overview:
            return sigma as tensor. When in iDDPM_edm mode, we need index as sigma.
        
        Arguments:
            sigma (:obj:`Union[torch.Tensor, float]`): Input sigma.
            return_index (:obj:`bool`): whether index is returned. Only iDDPM_edm type needs it.
            
        Returns:
            sigma (:obj:`torch.Tensor`): Output sigma in Tensor format.
        """
        if self.precondition_type == "iDDPM_edm":
            sigma = torch.as_tensor(sigma)
            index = torch.cdist(sigma.to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
            result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
            return result.reshape(sigma.shape)
        else:
            return torch.as_tensor(sigma)
        
    def get_precondition_c(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Overview:
            Obtain precondition c according to sigma including c_skip, c_out, c_in, c_noise 
            Accordig to section Network and preconditioning Table 1, 4 precondition functions are shown as follows:
                
            .. math::
                \mathbf{c}_{\mathrm{skip}}(\sigma), \mathbf{c}_{\mathrm{out}}(\sigma), \mathbf{c}_{\mathrm{in}}(\sigma), \mathbf{c}_{\mathrm{noise}}(\sigma)
        
        Arguments:
            sigma (:obj:`torch.Tensor`): Input sigma.
        
        Returns:
            c_skip (:obj:`torch.Tensor`): Output c_skip(sigma).
            c_out (:obj:`torch.Tensor`): Output c_out(sigma). 
            c_in (:obj:`torch.Tensor`): Output c_in(sigma).
            c_noise (:obj:`torch.Tensor`): Output c_noise(sigma).
        """
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
    
    def forward(self, sigma: Tensor, x: Tensor, condition: Optional[Tensor]=None, **model_kwargs):
        """
        Overview:
            Obtain denoiser from basic denoise network and precondition scaling functions, which is given as follows:
            
            .. math:
                \mathbf{D}_{\theta} (\mathbf{x}; \sigma; c) = \mathbf{c}_{\mathrm{skip}}(\sigma) \mathbf{x} + \mathbf{c}_{\mathrm{out}}(\sigma) \mathbf{F}_{\theta}(\mathbf{c}_{\mathrm{in}}(\sigma)\mathbf{x}; \mathbf{c}_{\mathrm{noise}}(\sigma); c)
        
        Arguments:
            sigma (:obj:`torch.Tensor`): Input sigma.
            x (:obj:`torch.Tensor`): Input x.
            condition: (:obj:`Optional[torch.Tensor]`): Input condition.
            
        Returns:
            D_x (:obj:`torch.Tensor`): Output denoiser.
        """
        # Suppose the first dim of x is batch size
        x = x.to(torch.float32)
        sigma_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        
        
        if sigma.numel() == 1:
            sigma = sigma.view(-1).expand(*sigma_shape)
        else:
            sigma = sigma.reshape(*sigma_shape)
        dtype = torch.float16 if (self.use_mixes_precision and x.device.type == 'cuda') else torch.float32
        c_skip, c_out, c_in, c_noise = self.get_precondition_c(sigma)
        F_x = self.denoise_model(c_noise.flatten(), (c_in * x).to(dtype), condition=condition, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x