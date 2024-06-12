import os
from typing import List
import torch
import torch.nn as nn


def sort_files_by_criteria(
    folder_path: str, start_string: str = "checkpoint_", end_string: str = ".pt"
) -> List[str]:
    """
    Overview:
        Sort the files in the specified folder by the criteria specified in the filename.
        If the filename is "checkpoint_N_M_..._1.pt", the files will be sorted in descending order by N, N-1, ..., 1.
    Arguments:
        - folder_path (:obj:`str`): The path to the folder containing the files.
    """

    files = os.listdir(folder_path)
    file_list = []

    for file in files:
        if file.startswith(start_string) and file.endswith(end_string):
            parts = file[len(start_string) : -len(end_string)].split(
                "_"
            )  # Split the filename by "_" and remove "checkpoint_" and ".pt"
            try:
                values = list(map(int, parts))  # Convert all parts to integers
                file_list.append(
                    tuple(reversed(values)) + (file,)
                )  # Append a tuple (N, N-1, ..., 1, filename) to the list
            except ValueError:
                pass  # Ignore files that don't match the expected pattern

    file_list.sort(reverse=True)  # Sort the list in descending order
    sorted_files = [
        file for values in file_list for file in [values[-1]]
    ]  # Extract the filenames from the sorted tuples
    return sorted_files


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def calculate_tensor_memory_size(tensor):
    memory_usage_in_bytes = tensor.element_size() * tensor.nelement()
    return memory_usage_in_bytes


def memory_allocated(device=torch.device("cuda")):
    return torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)
