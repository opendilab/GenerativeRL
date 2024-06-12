from typing import Any, Dict, List, Tuple, Union

import torch
import torchsde
from torch import nn


class TorchSDE(nn.Module):
    """
    Overview:
        The SDE class for torchsde library, wich is an object with methods `f` and `g` representing the drift and diffusion.
        The output of `g` should be a single tensor of size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m) for SDEs of other noise types,
        where d is the dimensionality of state and m is the dimensionality of Brownian motion.
    """

    def __init__(
        self,
        drift,
        diffusion,
        noise_type,
        sde_type,
    ):
        """
        Overview:
            Initialize the SDE object.
        Arguments:
            drift (:obj:`nn.Module`): The function that defines the drift of the SDE.
            diffusion (:obj:`nn.Module`): The function that defines the diffusion of the SDE.
            noise_type (:obj:`str`): The type of noise of the SDE. It can be 'diagonal', 'general', 'scalar' or 'additive'.
            sde_type (:obj:`str`): The type of the SDE. It can be 'ito' or 'stratonovich'.
        """
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

        self.noise_type = noise_type
        self.sde_type = sde_type

    def f(self, t, y):
        """
        Overview:
            The drift function of the SDE.
        """
        return self.drift(t, y)

    def g(self, t, y):
        """
        Overview:
            The diffusion function of the SDE.
        """
        return self.diffusion(t, y)


class SDESolver:

    def __init__(
        self,
        sde_solver="euler",
        sde_noise_type="diagonal",
        sde_type="ito",
        dt=0.001,
        atol=1e-5,
        rtol=1e-5,
        library="torchsde",
        **kwargs,
    ):
        """
        Overview:
            Initialize the SDE solver using torchsde library.
        Arguments:
            sde_solver (:obj:`str`): The SDE solver to use.
            sde_noise_type (:obj:`str`): The type of noise of the SDE. It can be 'diagonal', 'general', 'scalar' or 'additive'.
            sde_type (:obj:`str`): The type of the SDE. It can be 'ito' or 'stratonovich'.
            dt (:obj:`float`): The time step.
            atol (:obj:`float`): The absolute tolerance.
            rtol (:obj:`float`): The relative tolerance.
            library (:obj:`str`): The library to use for the ODE solver. Currently, it supports 'torchsde'.
            **kwargs: Additional arguments for the ODE solver.
        """
        super().__init__()
        self.sde_solver = sde_solver
        self.sde_noise_type = sde_noise_type
        self.sde_type = sde_type
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe_drift = 0
        self.nfe_diffusion = 0
        self.kwargs = kwargs
        self.library = library

    def integrate(self, drift, diffusion, x0, t_span, logqp=False, adaptive=False):
        """
        Overview:
            Integrate the SDE.
        Arguments:
            drift (:obj:`nn.Module`): The function that defines the ODE.
            diffusion (:obj:`nn.Module`): The function that defines the ODE.

        """

        batch_size = x0.shape[0]
        data_size = x0.shape[1:]

        self.nfe_drift = 0
        self.nfe_diffusion = 0

        def forward_drift(t, x):
            self.nfe_drift += 1
            x = x.reshape(batch_size, *data_size)
            f = drift(t, x)
            return f.reshape(batch_size, -1)

        def forward_diffusion(t, x):
            self.nfe_diffusion += 1
            x = x.reshape(batch_size, *data_size)
            g = diffusion(t, x)
            return g.reshape(batch_size, -1)

        sde = TorchSDE(
            drift=forward_drift,
            diffusion=forward_diffusion,
            noise_type=self.sde_noise_type,
            sde_type=self.sde_type,
        )

        x0 = x0.reshape(batch_size, -1)

        trajectory = torchsde.sdeint(
            sde,
            x0,
            t_span,
            method=self.sde_solver,
            dt=self.dt,
            rtol=self.rtol,
            atol=self.atol,
            logqp=logqp,
            adaptive=adaptive,
            **self.kwargs,
        )

        trajectory = trajectory.reshape(t_span.shape[0], batch_size, *data_size)

        return trajectory
