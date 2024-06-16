from typing import Union

import torch
import treetensor
from tensordict import TensorDict

from grl.numerical_methods.probability_path import (
    ConditionalProbabilityPath,
)


class StochasticProcess:
    """
    Overview:
        Class for describing a stochastic process for generative models.
    Interfaces:
        ``__init__``, ``mean``, ``std``, ``velocity``, ``direct_sample``, ``direct_sample_with_noise``, ``velocity_SchrodingerBridge``, ``score_SchrodingerBridge``
    """

    def __init__(self, path: ConditionalProbabilityPath, t_max: float = 1.0) -> None:

        super().__init__()
        self.path = path
        self.t_max = t_max

    def mean(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the mean of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        if isinstance(x0, torch.Tensor):
            if x0 is not None and len(x0.shape) > len(t.shape):
                t = t[(...,) + (None,) * (len(x0.shape) - len(t.shape))].expand(
                    x0.shape
                )
                return x0 * (1 - t) + x1 * t
            else:
                return x0 * (1 - t) + x1 * t
        elif isinstance(x0, treetensor.torch.Tensor):
            raise NotImplementedError("Not implemented yet")
        elif isinstance(x0, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x0)))

    def std(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the standard deviation of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        if isinstance(x0, torch.Tensor):
            if x0 is not None and len(x0.shape) > len(t.shape):
                return self.path.std(t)[
                    (...,) + (None,) * (len(x0.shape) - len(t.shape))
                ].expand(x0.shape)
            else:
                return self.path.std(t)
        elif isinstance(x0, treetensor.torch.Tensor):
            raise NotImplementedError("Not implemented yet")
        elif isinstance(x0, TensorDict):
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Invalid type of x: {}".format(type(x0)))

    def velocity(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the velocity of the state at time t given the state x.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        return x1 - x0

    def direct_sample(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Return the sample of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        """

        # TODO: make it compatible with TensorDict

        return self.mean(t, x0, x1, condition) + self.std(
            t, x0, x1, condition
        ) * torch.randn_like(x0).to(x0.device)

    def direct_sample_with_noise(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
        noise: Union[torch.Tensor, TensorDict] = None,
    ):
        """
        Overview:
            Return the sample of the state at time t given the initial state x0 and the final state x1 with noise.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The input noise.
        """
        return self.mean(t, x0, x1, condition) + self.std(
            t, x0, x1, condition
        ) * noise.to(x0.device)

    def velocity_SchrodingerBridge(
        self,
        t: torch.Tensor,
        x0: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        x1: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict] = None,
        noise: Union[torch.Tensor, TensorDict] = None,
    ):
        """
        Overview:
            Return the velocity of the state at time t given the state x with noise.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 0.
            x1 (:obj:`Union[torch.Tensor, TensorDict]`): The input state at time 1.
            condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
            noise (:obj:`Union[torch.Tensor, TensorDict]`): The input noise.
        """
        return (
            self.path.std_prime(t).unsqueeze(1) * self.std(t, x0, x1, condition) * noise
            + x1
            - x0
        )

    def score_SchrodingerBridge(self, t):
        return self.path.lambd(t)
