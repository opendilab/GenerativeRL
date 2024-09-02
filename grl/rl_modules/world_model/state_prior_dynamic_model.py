import torch

from easydict import EasyDict
from torch import nn

from grl.generative_models import get_generative_model

class StatePriorDynamicModel(nn.Module):
    """
    Overview:
        Dynamic model that use state as sampling prior.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        config: EasyDict,
    ):
        """
        Overview:
            Initialize the world model.
        Arguments:
            - config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()

        self.config = config
        self.model = get_generative_model(config.model_type)(config.model_config)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the next state given the current state and current condition.
            Condition usually is the action at the current time step or a combination of action and state in the past.
        Arguments:
            - state (:obj:`torch.Tensor`): The current state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.sample(x0=state, condition=condition)

    def sample(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the next state given the current state and current condition.
            Condition usually is the action at the current time step or a combination of action and state in the past.
        Arguments:
            - state (:obj:`torch.Tensor`): The current state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.sample(x0=state, condition=condition)
    
    def log_prob(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the log probability of the next state given the current state and current condition.
            Condition usually is the action at the current time step or a combination of action and state in the past.
        Arguments:
            - state (:obj:`torch.Tensor`): The current state.
            - next_state (:obj:`torch.Tensor`): The next state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.log_prob(x0=state, x1=next_state, condition=condition)

