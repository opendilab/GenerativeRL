from typing import Union

import torch
from easydict import EasyDict
from tensordict import TensorDict


class GaussianConditionalProbabilityPath:
    r"""
    Overview:
        Gaussian conditional probability path.

        General case:

        .. math::
            p(x(t)|x(0))=\mathcal{N}(x(t);\mu(t,x(0)),\sigma^2(t,x(0))I)

        If written in the form of SDE:

        .. math::
            \mathrm{d}x=f(x,t)\mathrm{d}t+g(t)w_{t}
        
        where :math:`f(x,t)` is the drift term, :math:`g(t)` is the diffusion term, and :math:`w_{t}` is the Wiener process.

        For diffusion model:

        .. math::
            p(x(t)|x(0))=\mathcal{N}(x(t);s(t)x(0),\sigma^2(t)I)
        
        or,

        .. math::
            p(x(t)|x(0))=\mathcal{N}(x(t);s(t)x(0),s^2(t)e^{-2\lambda(t)}I)
    
        If written in the form of SDE:

        .. math::
            \mathrm{d}x=\frac{s'(t)}{s(t)}x(t)\mathrm{d}t+s^2(t)\sqrt{2(\frac{s'(t)}{s(t)}-\lambda'(t))}e^{-\lambda(t)}\mathrm{d}w_{t}
        
        or,

        .. math::
            \mathrm{d}x=f(t)x(t)\mathrm{d}t+g(t)w_{t}
            

        where :math:`s(t)` is the scale factor, :math:`\sigma^2(t)I` is the covariance matrix, \
            :math:`\sigma(t)` is the standard deviation with the scale factor, \
            :math:`e^{-2\lambda(t)}I` is the covariance matrix without the scale factor, \
            :math:`\lambda(t)` is the half-logSNR, which is the difference between the log scale factor and the log standard deviation, \
            :math:`\lambda(t)=\log(s(t))-\log(\sigma(t))`.

        For VP SDE:

        .. math::
            p(x(t)|x(0))=\mathcal{N}(x(t);x(0)e^{-\frac{1}{2}\int_{0}^{t}{\beta(s)\mathrm{d}s}},(1-e^{-\int_{0}^{t}{\beta(s)\mathrm{d}s}})I)
    
        For Linear VP SDE:

        .. math::
            p(x(t)|x(0))=\mathcal{N}(x(t);x(0)e^{-\frac{\beta(1)-\beta(0)}{4}t^2-\frac{\beta(0)}{2}t},(1-e^{-\frac{\beta(1)-\beta(0)}{2}t^2-\beta(0)t})I)

        #TODO: add more details for
        Cosine VP SDE;
        General VE SDE;
        OPT-Flow;

    """

    def __init__(self, config: EasyDict) -> None:
        """
        Overview:
            Initialize the Gaussian conditional probability path.
        Arguments:
            config (:obj:`EasyDict`): The configuration of the Gaussian conditional probability path.
        """
        self.config = config
        self.type = config.type
        self.t_max = 1.0 if not hasattr(config, "t_max") else config.t_max
        assert self.type in [
            "diffusion",
            "vp_sde",
            "linear_vp_sde",
            "cosine_vp_sde",
            "general_ve_sde",
            "op_flow",
            "linear",
            "gvp",
        ], "Unknown type of Gaussian conditional probability path {}".format(type)

    def drift_coefficient(
        self,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Return the drift coefficient term of the Gaussian conditional probability path.
            The drift term is given by the following:

            .. math::
                f(t)

            which satisfies the following SDE:

            .. math::
                \mathrm{d}x=f(t)x(t)\mathrm{d}t+g(t)w_{t}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            drift (:obj:`Union[torch.Tensor, TensorDict]`): The output drift term.
        """

        if self.type == "linear_vp_sde":
            # TODO: make it compatible with TensorDict
            return -0.5 * (
                self.config.beta_0 + t * (self.config.beta_1 - self.config.beta_0)
            )
        elif self.type == "linear":
            return -torch.ones_like(t) / (1.0000001 - t)
        elif self.type == "gvp":
            return -0.5 * torch.pi * torch.tan(torch.pi * t / 2.0)
        else:
            raise NotImplementedError(
                "Drift coefficient term for type {} is not implemented".format(
                    self.type
                )
            )

    def drift(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Return the drift term of the Gaussian conditional probability path.
            The drift term is given by the following:

            .. math::
                f(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            drift (:obj:`Union[torch.Tensor, TensorDict]`): The output drift term.
        """

        if self.type == "linear_vp_sde":
            # TODO: make it compatible with TensorDict
            return torch.einsum("i...,i->i...", x, self.drift_coefficient(t))
        elif self.type == "linear":
            return torch.einsum("i...,i->i...", x, self.drift_coefficient(t))
        elif self.type == "gvp":
            return torch.einsum("i...,i->i...", x, self.drift_coefficient(t))
        else:
            raise NotImplementedError(
                "Drift term for type {} is not implemented".format(self.type)
            )

    def diffusion(
        self,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Return the diffusion term of the Gaussian conditional probability path.
            The diffusion term is given by the following:

            .. math::
                g(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            diffusion (:obj:`Union[torch.Tensor, TensorDict]`): The output diffusion term.
        """

        if self.type == "linear_vp_sde":
            return torch.sqrt(
                self.config.beta_0 + t * (self.config.beta_1 - self.config.beta_0)
            )
        elif self.type == "linear":
            return torch.sqrt(2 * t + 2 * t * t / (1.0000001 - t))
        elif self.type == "gvp":
            first = torch.pi * torch.sin(torch.pi * t * 0.5)
            second = (
                torch.sin(torch.pi * t * 0.5)
                * torch.sin(torch.pi * t * 0.5)
                * torch.tan(torch.pi * t * 0.5)
            )
            return torch.sqrt(first + second)
        else:
            raise NotImplementedError(
                "Diffusion term for type {} is not implemented".format(self.type)
            )

    def diffusion_squared(
        self,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Return the diffusion term of the Gaussian conditional probability path.
            The diffusion term is given by the following:

            .. math::
                g^2(x,t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            diffusion (:obj:`Union[torch.Tensor, TensorDict]`): The output diffusion term.
        """

        if self.type == "linear_vp_sde":
            return self.config.beta_0 + t * (self.config.beta_1 - self.config.beta_0)
        elif self.type == "linear":
            return 2 * t + 2 * t * t / (1.0000001 - t)
        elif self.type == "gvp":
            first = torch.pi * torch.sin(torch.pi * t * 0.5)
            second = (
                torch.sin(torch.pi * t * 0.5)
                * torch.sin(torch.pi * t * 0.5)
                * torch.tan(torch.pi * t * 0.5)
            )
            return first + second
        else:
            raise NotImplementedError(
                "Diffusion term for type {} is not implemented".format(self.type)
            )

    def scale(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the scale factor of the Gaussian conditional probability path, which is

            .. math::
                s(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            scale (:obj:`torch.Tensor`): The scale factor.
        """

        # TODO: implement the scale factor for other Gaussian conditional probability path

        if self.type == "linear_vp_sde":
            return torch.exp(
                -0.25 * t**2 * (self.config.beta_1 - self.config.beta_0)
                - 0.5 * t * self.config.beta_0
            )
        elif self.type == "linear":
            return 1 - t
        elif self.type == "gvp":
            return torch.cos(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Scale factor for type {} is not implemented".format(self.type)
            )

    def log_scale(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the log scale factor of the Gaussian conditional probability path, which is

            .. math::
                \log(s(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            log_scale (:obj:`torch.Tensor`): The log scale factor.
        """

        # TODO: implement the scale factor for other Gaussian conditional probability path

        if self.type == "linear_vp_sde":
            return (
                -0.25 * t**2 * (self.config.beta_1 - self.config.beta_0)
                - 0.5 * t * self.config.beta_0
            )
        elif self.type == "linear":
            return torch.log(1.0 - t)
        elif self.type == "gvp":
            return torch.log(torch.cos(0.5 * torch.pi * t))
        else:
            raise NotImplementedError(
                "Log scale factor for type {} is not implemented".format(self.type)
            )

    def d_log_scale_dt(
        self,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Compute the time derivative of the log scale factor of the Gaussian conditional probability path, which is

            .. math::
                \log(s'(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            d_log_scale_dt (:obj:`Union[torch.Tensor, TensorDict]`): The time derivative of the log scale factor.
        """

        if self.type == "linear_vp_sde":
            return -0.5 * t * (self.config.beta_1 - self.config.beta_0)
        elif self.type == "linear":
            return -1.0 / (1.0000001 - t)
        elif self.type == "gvp":
            return -0.5 * torch.pi * torch.tan(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Time derivative of the log scale factor for type {} is not implemented".format(
                    self.type
                )
            )

    def d_scale_dt(
        self,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, TensorDict]:
        r"""
        Overview:
            Compute the time derivative of the scale factor of the Gaussian conditional probability path, which is

            .. math::
                s'(t)

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            d_scale_dt (:obj:`Union[torch.Tensor, TensorDict]`): The time derivative of the scale factor.
        """

        if self.type == "linear_vp_sde":
            return -0.5 * t * (self.config.beta_1 - self.config.beta_0) * self.scale(t)
        elif self.type == "linear":
            return -1.0 * torch.ones_like(t, dtype=torch.float32)
        elif self.type == "gvp":
            return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Time derivative of the scale factor for type {} is not implemented".format(
                    self.type
                )
            )

    def std(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the standard deviation of the Gaussian conditional probability path, which is

            .. math::
                \sqrt{\Sigma(t)}

            or

            .. math::
                \sigma(t)

            or

            .. math::
                s(t)e^{-\lambda(t)}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            std (:obj:`torch.Tensor`): The standard deviation.
        """

        if self.type == "linear_vp_sde":
            return torch.sqrt(1.0 - self.scale(t) ** 2)
        elif self.type == "linear":
            return t
        elif self.type == "gvp":
            return torch.sin(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Standard deviation for type {} is not implemented".format(self.type)
            )

    def d_std_dt(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the time derivative of the standard deviation of the Gaussian conditional probability path, which is

            .. math::
                \frac{\mathrm{d}\sigma(t)}{\mathrm{d}t}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        """

        if self.type == "linear_vp_sde":
            return -self.d_scale_dt(t) * self.scale(t) / self.std(t)
        elif self.type == "linear":
            return torch.ones_like(t, dtype=torch.float32)
        elif self.type == "gvp":
            return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Time derivative of standard deviation for type {} is not implemented".format(
                    self.type
                )
            )

    def covariance(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the covariance matrix of the Gaussian conditional probability path, which is

            .. math::
                \Sigma(t)

            or

            .. math::
                \sigma^2(t)I

            or

            .. math::
                s^2(t)e^{-2\lambda(t)}I

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            covariance (:obj:`torch.Tensor`): The covariance matrix.
        """

        if self.type == "linear_vp_sde":
            return 1.0 - self.scale(t) ** 2
        elif self.type == "linear":
            return t * t
        elif self.type == "gvp":
            return torch.sin(0.5 * torch.pi * t) * torch.sin(0.5 * torch.pi * t)
        else:
            raise NotImplementedError(
                "Covariance for type {} is not implemented".format(self.type)
            )

    def d_covariance_dt(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the time derivative of the covariance matrix of the Gaussian conditional probability path, which is

            .. math::
                \frac{\mathrm{d}\Sigma(t)}{\mathrm{d}t}

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            d_covariance_dt (:obj:`torch.Tensor`): The time derivative of the covariance matrix.
        """

        if self.type == "linear_vp_sde":
            return -2.0 * self.scale(t) * self.d_scale_dt(t)
        elif self.type == "linear":
            return 2.0 * t
        elif self.type == "gvp":
            return (
                torch.pi * torch.sin(torch.pi * t * 0.5) * torch.cos(torch.pi * t * 0.5)
            )
        else:
            raise NotImplementedError(
                "Time derivative of covariance for type {} is not implemented".format(
                    self.type
                )
            )

    def HalfLogSNR(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the half-logSNR of the Gaussian conditional probability path, which is

            .. math::
                \log(s(t))-\log(\sigma(t))

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
        Returns:
            HalfLogSNR (:obj:`torch.Tensor`): The half-logSNR.
        """

        if self.type == "linear_vp_sde":
            return self.log_scale(t) - 0.5 * torch.log(1.0 - self.scale(t) ** 2)
        else:
            raise NotImplementedError(
                "Half-logSNR for type {} is not implemented".format(self.type)
            )

    def InverseHalfLogSNR(self, HalfLogSNR: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Compute the inverse function of the half-logSNR of the Gaussian conditional probability path.
            Since the half-logSNR is an invertible function, we can compute the time t from the half-logSNR.
            For linear VP SDE, the inverse function is

            .. math::
                t(\lambda)=\frac{1}{\beta_1-\beta_0}(\sqrt{\beta_0^2+2(\beta_1-\beta_0)\log{(e^{-2\lambda}+1)}}-\beta_0)

            or,

            .. math::
                t(\lambda)=\frac{2(\beta_1-\beta_0)\log{(e^{-2\lambda}+1)}}{\sqrt{\beta_0^2+2(\beta_1-\beta_0)\log{(e^{-2\lambda}+1)}}+\beta_0}

        Arguments:
            HalfLogSNR (:obj:`torch.Tensor`): The input half-logSNR.
        Returns:
            t (:obj:`torch.Tensor`): The time.
        """

        if self.type == "linear_vp_sde":
            numerator = 2.0 * torch.logaddexp(
                -2.0 * HalfLogSNR, torch.zeros((1,)).to(HalfLogSNR)
            )
            denominator = (
                torch.sqrt(
                    self.config.beta_0**2
                    + (self.config.beta_1 - self.config.beta_0) * numerator
                )
                + self.config.beta_0
            )
            return numerator / denominator
        else:
            raise NotImplementedError(
                "Inverse function of half-logSNR for type {} is not implemented".format(
                    self.type
                )
            )


class ConditionalProbabilityPath:
    """
    Overview:
        Conditional probability path for general continuous-time normalizing flow.

    """

    def __init__(self, config) -> None:
        self.config = config

    def std(self, t: torch.Tensor) -> torch.Tensor:

        return torch.tensor(self.config.sigma, device=t.device)


class SchrodingerBridgePath:
    def __init__(self, config) -> None:
        self.config = config

    def std(self, t: torch.Tensor) -> torch.Tensor:
        return self.config.sigma * torch.sqrt(t * (1 - t))

    def lambd(self, t: torch.Tensor) -> torch.Tensor:
        return 2 * self.std(t) / (self.config.sigma**2)

    def std_prime(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - 2 * t) / (2 * t * (1 - t))
