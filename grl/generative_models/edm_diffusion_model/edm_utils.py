import numpy as np
import torch
from easydict import EasyDict

############# Sampling Section #############

# Scheduling in Table 1 of paper https://arxiv.org/abs/2206.00364
SIGMA_T = {
    "VP_edm": lambda t, beta_d=19.9, beta_min=0.1: ((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1) ** 0.5,
    "VE_edm": lambda t, **kwargs: t.sqrt(),
    "iDDPM_edm": lambda t, **kwargs: t,
    "EDM": lambda t, **kwargs: t
}

SIGMA_T_DERIV = {
    "VP_edm": lambda t, beta_d=19.9, beta_min=0.1: 0.5 * (beta_min + beta_d * t) * (SIGMA_T["VP_edm"](t, beta_d, beta_min) + (1 / SIGMA_T["VP_edm"](t, beta_d, beta_min))),
    "VE_edm": lambda t, **kwargs: 1 / (2 * t.sqrt()),
    "iDDPM_edm": lambda t, **kwargs: 1,
    "EDM": lambda t, **kwargs: 1
}

SIGMA_T_INV = {
    "VP_edm": lambda sigma, beta_d=19.9, beta_min=0.1: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1)).log() - beta_min).sqrt() / beta_d,
    "VE_edm": lambda sigma, **kwargs: sigma ** 2,
    "iDDPM_edm": lambda sigma, **kwargs: sigma,
    "EDM": lambda sigma, **kwargs: sigma
}

# Scaling in Table 1
SCALE_T = {
    "VP_edm": lambda t, beta_d=19.9, beta_min=0.1: 1 / (1 + SIGMA_T["VP_edm"](t, beta_d, beta_min) ** 2).sqrt(),
    "VE_edm": lambda t, **kwargs: 1,
    "iDDPM_edm": lambda t, **kwargs: 1,
    "EDM": lambda t, **kwargs: 1
}

SCALE_T_DERIV = {
    "VP_edm": lambda t, beta_d=19.9, beta_min=0.1: -SIGMA_T["VP_edm"](t, beta_d, beta_min) * SIGMA_T_DERIV["VP_edm"](t, beta_d, beta_min) * (SCALE_T["VP_edm"](t, beta_d, beta_min) ** 3),
    "VE_edm": lambda t, **kwargs: 0,
    "iDDPM_edm": lambda t, **kwargs: 0,
    "EDM": lambda t, **kwargs: 0   
}


INITIAL_SIGMA_MIN = {
    "VP_edm": float(SIGMA_T["VP_edm"](torch.tensor(1e-3), 19.9, 0.1)), 
    "VE_edm": 0.02, 
    "iDDPM_edm": 0.002, 
    "EDM": 0.002
}

INITIAL_SIGMA_MAX = {
    "VP_edm": float(SIGMA_T["VP_edm"](torch.tensor(1.), 19.9, 0.1)), 
    "VE_edm": 100, 
    "iDDPM_edm": 81, 
    "EDM": 80
}

###### Default Params ######

DEFAULT_PARAM = EasyDict({
    "VP_edm":
    {
        "beta_d": 19.9,
        "beta_min": 0.1,
        "M": 1000,
        "epsilon_t": 1e-5,
    },
    "VE_edm":
    {
        "sigma_min": 0.02,
        "sigma_max": 100
    },
    "iDDPM_edm":
    {
        "C_1": 0.001,
        "C_2": 0.008,
        "M": 1000
    },
    "EDM":
    {
        "sigma_min": 0.002,
        "sigma_max": 80,
        "sigma_data": 0.5,
        "P_mean": -1.2,
        "P_std": 1.2
    }   
})

DEFAULT_SOLVER_PARAM = EasyDict(
    {
    "num_steps": 18,
    "epsilon_s": 1e-3,
    "rho": 7,
    "S_churn": 0.,
    "S_min": 0.,
    "S_max": float("inf"),
    "S_noise": 1.,
    "alpha": 1
})
