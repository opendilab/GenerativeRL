from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from easydict import EasyDict


def merge_dict1_into_dict2(
    dict1: Union[Dict, EasyDict], dict2: Union[Dict, EasyDict]
) -> Union[Dict, EasyDict]:
    """
    Overview:
        Merge two dictionaries recursively. \
        Update values in dict2 with values in dict1, and add new keys from dict1 to dict2.
    Arguments:
        - dict1 (:obj:`dict`): The first dictionary.
        - dict2 (:obj:`dict`): The second dictionary.
    """
    for key, value in dict1.items():
        if key in dict2 and isinstance(value, dict) and isinstance(dict2[key], dict):
            # Both values are dictionaries, so merge them recursively
            merge_dict1_into_dict2(value, dict2[key])
        else:
            # Either the key doesn't exist in dict2 or the values are not dictionaries
            dict2[key] = value

    return dict2


def merge_two_dicts_into_newone(
    dict1: Union[Dict, EasyDict], dict2: Union[Dict, EasyDict]
) -> Union[Dict, EasyDict]:
    """
    Overview:
        Merge two dictionaries recursively into a new dictionary. \
        Update values in dict2 with values in dict1, and add new keys from dict1 to dict2.
    Arguments:
        - dict1 (:obj:`dict`): The first dictionary.
        - dict2 (:obj:`dict`): The second dictionary.
    """
    dict2 = deepcopy(dict2)
    return merge_dict1_into_dict2(dict1, dict2)
