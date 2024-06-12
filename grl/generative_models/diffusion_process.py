from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict
from torch.distributions import Distribution

from grl.numerical_methods.ode import ODE
from grl.numerical_methods.probability_path import GaussianConditionalProbabilityPath
from grl.numerical_methods.sde import SDE


class DiffusionProcess:
    """
    Overview:
        Common methods of diffusion process.
    """

    def __init__(self, path: GaussianConditionalProbabilityPath, t_max: float = 1.0):
        """
        Overview:
            Initialize the diffusion process.
        Arguments:
            path (:obj:`GaussianConditionalProbabilityPath`): The Gaussian conditional probability path.
            t_max (:obj:`float`): The maximum time.
        """
        super().__init__()
        self.path = path
        self.t_max = t_max

    def drift(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the drift term of the diffusion process.
            The drift term is given by the following:

            .. math::
                f(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            drift (:obj:`Union[torch.Tensor, TensorDict]`): The output drift term.
        """

        if isinstance(x, torch.Tensor):
            if len(x.shape) > len(t.shape):
                return x * self.path.drift_coefficient(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return x * self.path.drift_coefficient(t)
        elif isinstance(x, treetensor.torch.Tensor):
            drift = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    drift[key] = value * self.path.drift_coefficient(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    drift[key] = value * self.path.drift_coefficient(t)
            return drift
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def drift_coefficient(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the drift coefficient of the diffusion process.
            The drift coefficient is given by the following:

            .. math::
                f(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            drift_coefficient (:obj:`Union[torch.Tensor, TensorDict]`): The output drift coefficient.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.drift_coefficient(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.drift_coefficient(t)
        elif isinstance(x, treetensor.torch.Tensor):
            drift_coefficient = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    drift_coefficient[key] = self.path.drift_coefficient(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    drift_coefficient[key] = self.path.drift_coefficient(t)
            return drift_coefficient
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def diffusion(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the diffusion term of the diffusion process.
            The diffusion term is given by the following:

            .. math::
                g(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            diffusion (:obj:`Union[torch.Tensor, TensorDict]`): The output diffusion term.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.diffusion(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.diffusion(t)
        elif isinstance(x, treetensor.torch.Tensor):
            diffusion = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    diffusion[key] = self.path.diffusion(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    diffusion[key] = self.path.diffusion(t)
            return diffusion
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def diffusion_squared(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the square of the diffusion term of the diffusion process.
            The square of the diffusion term is given by the following:

            .. math::
                g^2(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            diffusion_squared (:obj:`Union[torch.Tensor, TensorDict]`): The output square of the diffusion term.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.diffusion_squared(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.diffusion_squared(t)
        elif isinstance(x, treetensor.torch.Tensor):
            diffusion_squared = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    diffusion_squared[key] = self.path.diffusion_squared(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    diffusion_squared[key] = self.path.diffusion_squared(t)
            return diffusion_squared
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def scale(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the scale of the diffusion process.
            The scale is given by the following:

            .. math::
                s(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            scale (:obj:`Union[torch.Tensor, TensorDict]`): The output scale.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.scale(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.scale(t)
        elif isinstance(x, treetensor.torch.Tensor):
            scale = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    scale[key] = self.path.scale(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    scale[key] = self.path.scale(t)
            return scale
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def log_scale(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the log scale of the diffusion process.
            The log scale is given by the following:

            .. math::
                \log(s(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            log_scale (:obj:`Union[torch.Tensor, TensorDict]`): The output log scale.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.log_scale(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.log_scale(t)
        elif isinstance(x, treetensor.torch.Tensor):
            log_scale = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    log_scale[key] = self.path.log_scale(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    log_scale[key] = self.path.log_scale(t)
            return log_scale
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def std(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the standard deviation of the diffusion process.
            The standard deviation is given by the following:

            .. math::
                \sigma(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            std (:obj:`Union[torch.Tensor, TensorDict]`): The output standard deviation.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.std(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.std(t)
        elif isinstance(x, treetensor.torch.Tensor):
            std = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    std[key] = self.path.std(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    std[key] = self.path.std(t)
            return std
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def covariance(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the covariance of the diffusion process.
            The covariance is given by the following:

            .. math::
                \Sigma(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            covariance (:obj:`Union[torch.Tensor, TensorDict]`): The output covariance.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.covariance(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.covariance(t)
        elif isinstance(x, treetensor.torch.Tensor):
            covariance = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    covariance[key] = self.path.covariance(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    covariance[key] = self.path.covariance(t)
            return covariance
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def velocity(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
        noise: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the velocity of the diffusion process.
            The velocity is given by the following:

            .. math::
                v(t,x):=\frac{\mathrm{d}(x_t|x_0)}{\mathrm{d}t}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The input noise.
        """

        if isinstance(x, torch.Tensor):
            if noise is None:
                noise = torch.randn_like(x).to(x.device)
            if len(x.shape) > len(t.shape):
                d_scale_dt = self.path.d_scale_dt(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
                d_std_dt = self.path.d_std_dt(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
                return d_scale_dt * x + d_std_dt * noise
            else:
                return self.path.d_scale_dt(t) * x + self.path.d_std_dt(t) * noise
        elif isinstance(x, treetensor.torch.Tensor):
            # TODO: check if is correct
            velocity = treetensor.torch.Tensor({}, device=t.device)
            if noise is None:
                for key, value in x.items():
                    if len(value.shape) > len(t.shape):
                        d_scale_dt = self.path.d_scale_dt(t)[
                            (...,) + (None,) * (len(value.shape) - len(t.shape))
                        ].expand(value.shape)
                        d_std_dt = self.path.d_std_dt(t)[
                            (...,) + (None,) * (len(value.shape) - len(t.shape))
                        ].expand(value.shape)
                        velocity[key] = (
                            d_scale_dt * value
                            + d_std_dt * torch.randn_like(value).to(value.device)
                        )
                    else:
                        velocity[key] = self.path.d_scale_dt(
                            t
                        ) * x + self.path.d_std_dt(t) * torch.randn_like(value).to(
                            value.device
                        )
            else:
                for key, value in x.items():
                    if len(value.shape) > len(t.shape):
                        d_scale_dt = self.path.d_scale_dt(t)[
                            (...,) + (None,) * (len(value.shape) - len(t.shape))
                        ].expand(value.shape)
                        d_std_dt = self.path.d_std_dt(t)[
                            (...,) + (None,) * (len(value.shape) - len(t.shape))
                        ].expand(value.shape)
                        velocity[key] = d_scale_dt * value + d_std_dt * noise[key]
                    else:
                        velocity[key] = (
                            self.path.d_scale_dt(t) * x
                            + self.path.d_std_dt(t) * noise[key]
                        )
            return velocity
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def HalfLogSNR(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the half log signal-to-noise ratio of the diffusion process.
            The half log signal-to-noise ratio is given by the following:

            .. math::
                \log(s(t))-\log(\sigma(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            HalfLogSNR (:obj:`torch.Tensor`): The half-logSNR.
        """

        if isinstance(x, torch.Tensor) or x is None:
            if x is not None and len(x.shape) > len(t.shape):
                return self.path.HalfLogSNR(t)[
                    (...,) + (None,) * (len(x.shape) - len(t.shape))
                ].expand(x.shape)
            else:
                return self.path.HalfLogSNR(t)
        elif isinstance(x, treetensor.torch.Tensor):
            HalfLogSNR = treetensor.torch.Tensor({}, device=t.device)
            for key, value in x.items():
                if len(value.shape) > len(t.shape):
                    HalfLogSNR[key] = self.path.HalfLogSNR(t)[
                        (...,) + (None,) * (len(value.shape) - len(t.shape))
                    ].expand(value.shape)
                else:
                    HalfLogSNR[key] = self.path.HalfLogSNR(t)
            return HalfLogSNR
        elif isinstance(x, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x)))

    def InverseHalfLogSNR(
        self,
        HalfLogSNR: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the inverse function of half log signal-to-noise ratio of the diffusion process, \
            which is the time at which the half log signal-to-noise ratio is given.
        Arguments:
            HalfLogSNR (:obj:`torch.Tensor`): The input half-logSNR.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            t (:obj:`torch.Tensor`): The output time.
        """

        if x is not None:
            if isinstance(x, torch.Tensor) or isinstance(x, TensorDict):
                return self.path.InverseHalfLogSNR(HalfLogSNR).to(x.device)
            elif isinstance(x, treetensor.torch.Tensor):
                return self.path.InverseHalfLogSNR(HalfLogSNR).to(
                    list(x.values())[0].device
                )
            else:
                raise ValueError("Invalid type of x: {}".format(type(x)))
        else:
            return self.path.InverseHalfLogSNR(HalfLogSNR)

    def sde(self, condition: Union[torch.Tensor, TensorDict] = None) -> SDE:
        """
        Overview:
            Return the SDE of diffusion process with the input condition.
        Arguments:
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sde (:obj:`SDE`): The SDE of diffusion process.
        """

        def sde_drift(
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
            return self.drift(t, x, condition)

        def sde_diffusion(
            t: torch.Tensor,
            x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
            return self.diffusion(t, x, condition)

        return SDE(drift=sde_drift, diffusion=sde_diffusion)

    def forward_sde(
        self,
        function: Union[Callable, nn.Module],
        function_type: str,
        forward_diffusion_function: Union[Callable, nn.Module] = None,
        forward_diffusion_squared_function: Union[Callable, nn.Module] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        T: torch.Tensor = torch.tensor(1.0),
    ) -> SDE:
        """
        Overview:
            Return the forward of equivalent reversed time SDE of the diffusion process with the input condition.
        Arguments:
            function (:obj:`Union[Callable, nn.Module]`): The input function.
            function_type (:obj:`str`): The type of the function.
            reverse_diffusion_function (:obj:`Union[Callable, nn.Module]`): The input reverse diffusion function.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            T (:obj:`torch.Tensor`): The maximum time.
        Returns:
            reverse_sde (:obj:`SDE`): The reverse diffusion process.
        """

        # TODO: validate these functions

        if function_type == "score_function":

            def sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return self.drift(t, x, condition) - 0.5 * (
                    self.diffusion_squared(t, x, condition)
                    - forward_diffusion_squared_function(t, x, condition)
                ) * function(t, x, condition)

            def sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return forward_diffusion_function(t, x, condition)

            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "noise_function":

            def sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return self.drift(t, x, condition) + 0.5 * (
                    self.diffusion_squared(t, x, condition)
                    - forward_diffusion_squared_function(t, x, condition)
                ) * function(t, x, condition) / self.std(t, x, condition)

            def sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return forward_diffusion_function(t, x, condition)

            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "velocity_function":

            def sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                v = function(t, x, condition)
                r = forward_diffusion_squared_function(t, x, condition) / (
                    self.diffusion_squared(t, x, condition) + 1e-8
                )
                return v - (v - self.drift(t, x, condition)) * r

            def sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return forward_diffusion_function(t, x, condition)

            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        elif function_type == "data_prediction_function":

            def sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                D = (
                    0.5
                    * (
                        self.diffusion_squared(t, x, condition)
                        - forward_diffusion_squared_function(t, x, condition)
                    )
                    / self.covariance(t, x, condition)
                )
                return (self.drift_coefficient(t, x) + D) * x - self.scale(
                    t, x, condition
                ) * D * function(t, x, condition)

            def sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return forward_diffusion_function(t, x, condition)

            return SDE(drift=sde_drift, diffusion=sde_diffusion)

        else:
            raise NotImplementedError(
                "Unknown type of function {}".format(function_type)
            )

    def forward_ode(
        self,
        function: Union[Callable, nn.Module],
        function_type: str,
        condition: Union[torch.Tensor, TensorDict] = None,
        T: torch.Tensor = torch.tensor(1.0),
    ) -> ODE:
        """
        Overview:
            Return the forward of equivalent reversed time ODE of the diffusion process with the input condition.
        """

        # TODO: validate these functions

        if function_type == "score_function":

            def ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return self.drift(t, x, condition) - 0.5 * self.diffusion_squared(
                    t, x, condition
                ) * function(t, x, condition)

            return ODE(drift=ode_drift)

        elif function_type == "noise_function":

            def ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return self.drift(t, x, condition) + 0.5 * self.diffusion_squared(
                    t, x, condition
                ) * function(t, x, condition) / self.std(t, x, condition)

            return ODE(drift=ode_drift)

        elif function_type == "velocity_function":

            def ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return function(t, x, condition)

            return ODE(drift=ode_drift)

        elif function_type == "data_prediction_function":

            def ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                D = (
                    0.5
                    * self.diffusion_squared(t, x, condition)
                    / self.covariance(t, x, condition)
                )
                return (self.drift_coefficient(t, x) + D) * x - self.scale(
                    t, x, condition
                ) * D * function(t, x, condition)

            return ODE(drift=ode_drift)

        else:
            raise NotImplementedError(
                "Unknown type of function {}".format(function_type)
            )

    def reverse_sde(
        self,
        function: Union[Callable, nn.Module],
        function_type: str,
        reverse_diffusion_function: Union[Callable, nn.Module] = None,
        reverse_diffusion_squared_function: Union[Callable, nn.Module] = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        T: torch.Tensor = torch.tensor(1.0),
    ) -> SDE:
        """
        Overview:
            Return the reversed time SDE of the diffusion process with the input condition.
        Arguments:
            function (:obj:`Union[Callable, nn.Module]`): The input function.
            function_type (:obj:`str`): The type of the function.
            reverse_diffusion_function (:obj:`Union[Callable, nn.Module]`): The input reverse diffusion function.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            T (:obj:`torch.Tensor`): The maximum time.
        Returns:
            reverse_sde (:obj:`SDE`): The reverse diffusion process.
        """

        if function_type == "score_function":

            def reverse_sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return -self.drift(T - t, x, condition) + 0.5 * (
                    self.diffusion_squared(T - t, x, condition)
                    + reverse_diffusion_squared_function(T - t, x, condition)
                ) * function(T - t, x, condition)

            def reverse_sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return reverse_diffusion_function(T - t, x, condition)

            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "noise_function":

            def reverse_sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return -self.drift(T - t, x, condition) - 0.5 * (
                    self.diffusion_squared(T - t, x, condition)
                    + reverse_diffusion_squared_function(T - t, x, condition)
                ) * function(T - t, x, condition) / self.std(T - t, x, condition)

            def reverse_sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return reverse_diffusion_function(T - t, x, condition)

            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "velocity_function":

            def reverse_sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                v = function(T - t, x, condition)
                r = reverse_diffusion_squared_function(T - t, x, condition) / (
                    self.diffusion_squared(T - t, x, condition) + 1e-8
                )
                return -v - (v - self.drift(T - t, x, condition)) * r

            def reverse_sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return reverse_diffusion_function(T - t, x, condition)

            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        elif function_type == "data_prediction_function":

            def reverse_sde_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                D = (
                    0.5
                    * (
                        self.diffusion_squared(T - t, x, condition)
                        + reverse_diffusion_squared_function(T - t, x, condition)
                    )
                    / self.covariance(T - t, x, condition)
                )
                return -(self.drift_coefficient(T - t, x) + D) * x + self.scale(
                    T - t, x, condition
                ) * D * function(T - t, x, condition)

            def reverse_sde_diffusion(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return reverse_diffusion_function(T - t, x, condition)

            return SDE(drift=reverse_sde_drift, diffusion=reverse_sde_diffusion)

        else:
            raise NotImplementedError(
                "Unknown type of function {}".format(function_type)
            )

    def reverse_ode(
        self,
        function: Union[Callable, nn.Module],
        function_type: str,
        condition: Union[torch.Tensor, TensorDict] = None,
        T: torch.Tensor = torch.tensor(1.0),
    ) -> ODE:
        """
        Overview:
            Return the reversed time ODE of the diffusion process with the input condition.
        """

        if function_type == "score_function":

            def reverse_ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return -self.drift(T - t, x, condition) + 0.5 * self.diffusion_squared(
                    T - t, x, condition
                ) * function(T - t, x, condition)

            return ODE(drift=reverse_ode_drift)

        elif function_type == "noise_function":

            def reverse_ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return -self.drift(T - t, x, condition) - 0.5 * self.diffusion_squared(
                    T - t, x, condition
                ) * function(T - t, x, condition) / self.std(T - t, x, condition)

            return ODE(drift=reverse_ode_drift)

        elif function_type == "velocity_function":

            def reverse_ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                return -function(T - t, x, condition)

            return ODE(drift=reverse_ode_drift)

        elif function_type == "data_prediction_function":

            def reverse_ode_drift(
                t: torch.Tensor,
                x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
            ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
                D = (
                    0.5
                    * self.diffusion_squared(T - t, x, condition)
                    / self.covariance(T - t, x, condition)
                )
                return -(self.drift_coefficient(T - t, x) + D) * x + self.scale(
                    T - t, x, condition
                ) * D * function(T - t, x, condition)

            return ODE(drift=reverse_ode_drift)

        else:
            raise NotImplementedError(
                "Unknown type of function {}".format(function_type)
            )

    def sample(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the sample of the state at time t given the initial state.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
        """
        return self.forward(x, t, condition).sample()

    def direct_sample(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of diffusion process.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        # TODO: make it compatible with TensorDict

        return self.scale(t, x) * x + self.std(t, x) * torch.randn_like(x).to(x.device)

    def direct_sample_and_return_noise(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the sample of the state at time t given the initial state by using gaussian conditional probability path of diffusion process and the noise.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        Returns:
            sample (:obj:`Union[torch.Tensor, TensorDict]`): The output sample.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The output noise.
        """

        noise = torch.randn_like(x).to(x.device)

        return self.scale(t, x) * x + self.std(t, x) * noise, noise
