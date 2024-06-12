import os
import random

import numpy as np
import torch


def set_seed(seed_value=None, cudnn_deterministic=True, cudnn_benchmark=False):
    """
    Overview:
        Set the random seed. If no seed value is provided, generate a random seed.
    Arguments:
        - seed_value (:obj:`int`, optional): The random seed to set. If None, a random seed will be generated.
        - cudnn_deterministic (:obj:`bool`, optional): Whether to make cuDNN operations deterministic. Defaults to True.
        - cudnn_benchmark (:obj:`bool`, optional): Whether to enable cuDNN benchmarking for convolutional operations. Defaults to False.
    Returns:
        - seed_value (:obj:`int`): The seed value used.
    """
    if seed_value is None:
        # Generate a random seed from system randomness
        seed_value = int.from_bytes(os.urandom(4), "little")

    random.seed(seed_value)  # Set seed for Python's built-in random library
    np.random.seed(seed_value)  # Set seed for NumPy
    torch.manual_seed(seed_value)  # Set seed for PyTorch
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Set PyTorch cuDNN behavior
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

    return seed_value


from .config import merge_dict1_into_dict2, merge_two_dicts_into_newone
from .log import log
from .statistics import find_parameters
