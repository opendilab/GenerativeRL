import torch

from easydict import EasyDict
from torch import nn

from grl.generative_models import get_generative_model

class DynamicModel(nn.Module):
    """
    Overview:
        General dynamic model.
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
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the next state given the current condition.
            Condition usually is a combination of action and state at the current time step or in the past.
        Arguments:
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.sample(condition=condition)

    def sample(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the next state given the current condition.
            Condition usually is a combination of action and state at the current time step or in the past.
        Arguments:
            - state (:obj:`torch.Tensor`): The current state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.sample(condition=condition)
    
    def log_prob(
        self,
        next_state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the log probability of the next state given current condition.
            Condition usually is a combination of action and state at the current time step or in the past.
        Arguments:
            - next_state (:obj:`torch.Tensor`): The next state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        return self.model.log_prob(x=next_state, condition=condition)

    def dynamic_loss(
        self,
        next_state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overview:
            Return the dynamic loss of the next state given current condition.
            Condition usually is a combination of action and state at the current time step or in the past.
        Arguments:
            - next_state (:obj:`torch.Tensor`): The next state.
            - condition (:obj:`torch.Tensor`): The condition.
        """

        if self.config.loss_type == "score_matching":
            return self.model.score_matching_loss(x=next_state, condition=condition)
        elif self.config.loss_type == "flow_matching":
            return self.model.flow_matching_loss(x=next_state, condition=condition)
        else:
            raise ValueError("Invalid loss type: {}".format(self.config.loss_type))
