from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from grl.numerical_methods.monte_carlo import MonteCarloSampler


class ScoreFunction:
    """
    Overview:
        Model of Score function in diffusion model.
    Interfaces:
        ``__init__``, ``forward``, ``score_matching_loss``
    """

    def __init__(
        self,
        model_type: str,
        process: object,
    ):
        """
        Overview:
            Initialize the score function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`object`): The process.
        """

        self.model_type = model_type
        self.process = process
        # TODO: add more types
        assert self.model_type in [
            "data_prediction_function",
            "noise_function",
            "score_function",
            "velocity_function",
            "denoiser_function",
        ], "Unknown type of ScoreFunction {}".format(type)

    def forward(
        self,
        model: Union[Callable, nn.Module],
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return score function of the model at time t given the initial state, which is the gradient of the log-likelihood.
            .. math::
                \nabla_{x_t} \log p_{\theta}(x_t)

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if self.model_type == "noise_function":
            return -model(t, x, condition) / self.process.std(t, x)
        elif self.model_type == "score_function":
            return model(t, x, condition)
        elif self.model_type == "velocity_function":
            # TODO: check if is correct
            return (
                -(model(t, x, condition) - self.process.drift(t, x))
                * 2.0
                / self.process.diffusion_squared(t, x)
            )
        elif self.model_type == "data_prediction_function":
            # TODO: check if is correct
            return -(
                x - self.process.scale(t, x) * model(t, x, condition)
            ) / self.process.covariance(t, x)
        else:
            raise NotImplementedError("Unknown type of ScoreFunction {}".format(type))

    def score_matching_loss(
        self,
        model: Union[Callable, nn.Module],
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        gaussian_generator: Callable = None,
        weighting_scheme: str = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the score matching loss function of the model given the initial state and the condition.

            .. math::
                \mathbb{E}_{t \sim p(t)} \left[ \mathbb{E}_{x_t \sim p(x_t | x_0)} \left[ \lambda(t) \left\| \nabla_{x_t} \log p_{\theta}(x_t) - \nabla_{x_t} \log p_{\theta}(x_t|x_0) \right\|^2 \right] \right]

            which is equivalent to

            .. math::
                \mathbb{E}_{t \sim p(t)} \left[ \mathbb{E}_{x_t \sim p(x_t | x_0)} \left[ \lambda(t) \left\| \nabla_{x_t} \log p_{\theta}(x_t) - \frac{x_t - x_0}{\sigma^2} \right\|^2 \right] \right]

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The intrinsic model of generative model, which can be a neural network representing the score function, noise function, velocity function, or data prediction function.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            - gaussian_generator (:obj:`Callable`): The Gaussian generator, which is a function that generates Gaussian noise of the same shape as x.
            - weighting_scheme (:obj:`str`): The weighting scheme for score matching loss, which can be "maximum_likelihood" or "vanilla".

            ..note::
                - "maximum_likelihood": The weighting scheme is based on the maximum likelihood estimation. Refer to the paper "Maximum Likelihood Training of Score-Based Diffusion Models" for more details. The weight :math:`\lambda(t)` is denoted as:

                    .. math::
                        \lambda(t) = g^2(t)

                    for numerical stability, we use Monte Carlo sampling to approximate the integral of :math:`\lambda(t)`.

                    .. math::
                        \lambda(t) = g^2(t) = p(t)\sigma^2(t)

                - "vanilla": The weighting scheme is based on the vanilla score matching, which balances the MSE loss by scaling the model output to the noise value. Refer to the paper "Score-Based Generative Modeling through Stochastic Differential Equations" for more details. The weight :math:`\lambda(t)` is denoted as:

                    .. math::
                        \lambda(t) = \sigma^2(t)
        """

        def get_random_t_samples(batch_size, device):
            if weighting_scheme is None or weighting_scheme == "maximum_likelihood":
                if not hasattr(self, "monte_carlo_sampler"):

                    def unnormalized_pdf(t):
                        return self.process.diffusion_squared(t) / (
                            self.process.covariance(t) + 1e-6
                        )

                    self.monte_carlo_sampler = MonteCarloSampler(
                        unnormalized_pdf=unnormalized_pdf,
                        x_min=0.0,
                        x_max=self.process.t_max,
                        device=device,
                    )
                return self.monte_carlo_sampler.sample(batch_size).to(device=device)
            elif weighting_scheme == "vanilla":
                # TODO: test esp
                eps = 1e-5
                return (
                    torch.rand(batch_size, device=device) * (self.process.t_max - eps)
                    + eps
                )
            else:
                raise NotImplementedError(
                    "Unknown weighting scheme {}".format(weighting_scheme)
                )

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

        def get_loss(noise_value, noise):
            if isinstance(noise_value, torch.Tensor):
                if average:
                    return torch.mean(
                        torch.sum(0.5 * (noise_value - noise) ** 2, dim=(1,))
                    )
                else:
                    return torch.sum(0.5 * (noise_value - noise) ** 2, dim=(1,))
            elif isinstance(noise_value, TensorDict):
                raise NotImplementedError("Not implemented yet")
            elif isinstance(noise_value, treetensor.torch.Tensor):
                if average:
                    return treetensor.torch.mean(
                        treetensor.torch.sum(
                            0.5 * (noise_value - noise) * (noise_value - noise),
                            dim=(1,),
                        )
                    )
                else:
                    return treetensor.torch.sum(
                        0.5 * (noise_value - noise) * (noise_value - noise), dim=(1,)
                    )
            else:
                raise NotImplementedError("Unknown type of noise_value {}".format(type))

        # TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            batch_size, device = get_batch_size_and_device(x)
            t_random = get_random_t_samples(batch_size, device)
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            x_t = (
                self.process.scale(t_random, x) * x
                + self.process.std(t_random, x) * noise
            )
            noise_value = model(t_random, x_t, condition=condition)
            loss = get_loss(noise_value, noise)
            return loss
        elif self.model_type == "score_function":
            batch_size, device = get_batch_size_and_device(x)
            t_random = get_random_t_samples(batch_size, device)
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            score_value = model(t_random, x_t, condition=condition)
            loss = get_loss(score_value * std, noise)
            return loss
        elif self.model_type == "velocity_function":
            batch_size, device = get_batch_size_and_device(x)
            t_random = get_random_t_samples(batch_size, device)
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            noise_value = (
                (velocity_value - self.process.drift(t_random, x_t))
                * 2.0
                * std
                / self.process.diffusion_squared(t_random, x_t)
            )
            loss = get_loss(noise_value, noise)
            return loss
        elif self.model_type == "data_prediction_function":
            batch_size, device = get_batch_size_and_device(x)
            t_random = get_random_t_samples(batch_size, device)
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            scale = self.process.scale(t_random, x)
            x_t = scale * x + std * noise
            data_predicted = model(t_random, x_t, condition=condition)
            noise_value = (x_t - scale * data_predicted) / std
            loss = get_loss(noise_value, noise)
            return loss
        else:
            raise NotImplementedError("Unknown type of score function {}".format(type))
