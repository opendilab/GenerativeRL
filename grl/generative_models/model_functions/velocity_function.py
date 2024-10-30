from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from grl.generative_models.diffusion_process import DiffusionProcess


class VelocityFunction:
    """
    Overview:
        Velocity function in diffusion model.
    Interfaces:
        ``__init__``, ``forward``, ``flow_matching_loss``
    """

    def __init__(
        self,
        model_type: str,
        process: DiffusionProcess,
    ):
        """
        Overview:
            Initialize the velocity function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`DiffusionProcess`): The process.
        """
        self.model_type = model_type
        self.process = process

    def forward(
        self,
        model: Union[Callable, nn.Module],
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return velocity of the model at time t given the initial state.
            .. math::
                v_{\theta}(t, x)
        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state at time t.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if self.model_type == "noise_function":
            # TODO: check if this is correct
            return self.process.drift(t, x) + 0.5 * self.process.diffusion_squared(
                t, x
            ) / self.process.std(t, x) * model(t, x, condition)
        elif self.model_type == "score_function":
            # TODO: check if this is correct
            return self.process.drift(t, x) - 0.5 * self.process.diffusion_squared(
                t, x
            ) * model(t, x, condition)
        elif self.model_type == "velocity_function":
            return model(t, x, condition)
        elif self.model_type == "data_prediction_function":
            # TODO: check if this is correct
            D = (
                0.5
                * self.process.diffusion_squared(t, x)
                / self.process.covariance(t, x)
            )
            return (self.process.drift_coefficient(t) + D) - D * self.process.scale(
                t
            ) * model(t, x, condition)
        else:
            raise NotImplementedError(
                "Unknown type of Velocity Function {}".format(type)
            )

    def flow_matching_loss(
        self,
        model: Union[Callable, nn.Module],
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        gaussian_generator: Callable = None,
        average: bool = True,
    ) -> torch.Tensor:

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

        def get_loss(velocity_value, velocity):
            if isinstance(velocity_value, torch.Tensor):
                if average:
                    return torch.mean(
                        torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
                    )
                else:
                    return torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
            elif isinstance(velocity_value, TensorDict):
                raise NotImplementedError("Not implemented yet")
            elif isinstance(velocity_value, treetensor.torch.Tensor):
                if average:
                    return treetensor.torch.mean(
                        treetensor.torch.sum(
                            0.5
                            * (velocity_value - velocity)
                            * (velocity_value - velocity),
                            dim=(1,),
                        )
                    )
                else:
                    return treetensor.torch.sum(
                        0.5 * (velocity_value - velocity) * (velocity_value - velocity),
                        dim=(1,),
                    )
            else:
                raise NotImplementedError(
                    "Unknown type of velocity_value {}".format(type)
                )

        # TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = (
                torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            )
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = (
                self.process.drift(t_random, x_t)
                + 0.5
                * self.process.diffusion_squared(t_random, x_t)
                * model(t_random, x_t, condition=condition)
                / std
            )
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "score_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = (
                torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            )
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = self.process.drift(
                t_random, x_t
            ) - 0.5 * self.process.diffusion_squared(t_random, x_t) * model(
                t_random, x_t, condition=condition
            )
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "velocity_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = (
                torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            )
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            velocity_value = model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "data_prediction_function":
            # TODO: check if this is correct
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x)
            t_random = (
                torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
            )
            if gaussian_generator is None:
                noise = torch.randn_like(x).to(device)
            else:
                noise = gaussian_generator(batch_size)
            std = self.process.std(t_random, x)
            x_t = self.process.scale(t_random, x) * x + std * noise
            D = (
                0.5
                * self.process.diffusion_squared(t_random, x)
                / self.process.covariance(t_random, x)
            )
            velocity_value = (
                self.process.drift_coefficient(t_random, x_t) + D
            ) * x_t - D * self.process.scale(t_random, x_t) * model(
                t_random, x_t, condition=condition
            )
            velocity = self.process.velocity(t_random, x, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        else:
            raise NotImplementedError(
                "Unknown type of velocity function {}".format(type)
            )

    def flow_matching_loss_icfm(
        self,
        model: Union[Callable, nn.Module],
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        average: bool = True,
        sum_all_elements: bool = True,
    ) -> torch.Tensor:

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

        def get_loss(velocity_value, velocity):
            if isinstance(velocity_value, torch.Tensor):
                if average:
                    return torch.mean(
                        torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
                    )
                else:
                    if sum_all_elements:
                        return torch.sum(
                            0.5 * (velocity_value - velocity) ** 2, dim=(1,)
                        )
                    else:
                        return 0.5 * (velocity_value - velocity) ** 2
            elif isinstance(velocity_value, TensorDict):
                raise NotImplementedError("Not implemented yet")
            elif isinstance(velocity_value, treetensor.torch.Tensor):
                if average:
                    return treetensor.torch.mean(
                        treetensor.torch.sum(
                            0.5
                            * (velocity_value - velocity)
                            * (velocity_value - velocity),
                            dim=(1,),
                        )
                    )
                else:
                    if sum_all_elements:
                        return treetensor.torch.sum(
                            0.5
                            * (velocity_value - velocity)
                            * (velocity_value - velocity),
                            dim=(1,),
                        )
                    else:
                        return (
                            0.5
                            * (velocity_value - velocity)
                            * (velocity_value - velocity)
                        )
            else:
                raise NotImplementedError(
                    "Unknown type of velocity_value {}".format(type)
                )

        # TODO: make it compatible with TensorDict
        if self.model_type == "noise_function":
            raise NotImplementedError("Not implemented yet")
        elif self.model_type == "score_function":
            raise NotImplementedError("Not implemented yet")
        elif self.model_type == "velocity_function":
            batch_size, device = get_batch_size_and_device(x0)
            t_random = torch.rand(batch_size, device=device) * self.process.t_max
            x_t = self.process.direct_sample(t_random, x0, x1)
            velocity_value = model(t_random, x_t, condition=condition)
            velocity = self.process.velocity(t_random, x0, x1)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "data_prediction_function":
            raise NotImplementedError("Not implemented yet")
        else:
            raise NotImplementedError(
                "Unknown type of velocity function {}".format(type)
            )
