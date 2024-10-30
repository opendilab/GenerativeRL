from typing import Dict
import torch
import numpy as np
from tensordict import TensorDict


def obs_transform(obs, device):

    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(device)
    elif isinstance(obs, Dict):
        obs = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
    elif isinstance(obs, torch.Tensor):
        obs = obs.float().to(device)
    elif isinstance(obs, TensorDict):
        obs = obs.to(device)
    else:
        raise ValueError("observation must be a dict, torch.Tensor, or np.ndarray")

    return obs


def action_transform(action, return_as_torch_tensor: bool = False):
    if isinstance(action, Dict):
        if return_as_torch_tensor:
            action = {k: v.cpu() for k, v in action.items()}
        else:
            action = {k: v.cpu().numpy() for k, v in action.items()}
    elif isinstance(action, torch.Tensor):
        if return_as_torch_tensor:
            action = action.cpu()
        else:
            action = action.numpy()
    elif isinstance(action, np.ndarray):
        pass
    else:
        raise ValueError("action must be a dict, torch.Tensor, or np.ndarray")

    return action


from .base import BaseAgent
from .qgpo import QGPOAgent
from .srpo import SRPOAgent
from .gm import GPAgent
from .idql import IDQLAgent
