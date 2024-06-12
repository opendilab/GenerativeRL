from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import treetensor
from tensordict import TensorDict


def gaussian_random_variable(
    data_size: Union[
        torch.Tensor, torch.Size, int, Tuple[int], List[int], Dict[Any, Any]
    ],
    device: torch.device = torch.device("cpu"),
    use_tree_tensor: bool = False,
) -> Callable:
    """
    Overview:
        A random Gaussian tensor generator.
    Arguments:
        data_size (:obj:`Union[int, Tuple[int], List[int], Dict[Any, Any]]`): The data size.
        device (:obj:`torch.device`): The device.
    Returns:
        generate (:obj:`Callable`): The random Gaussian tensor generator.
    .. note::
        If data_size is a dictionary, the return value will be a (:obj:`TensorDict`) or (:obj:`treetensor.torch.Tensor`).

    Examples:
        >>> print(gaussian_random_variable(3)())
        >>> print(gaussian_random_variable((3, 4))())
        >>> print(gaussian_random_variable({"a": 3, "b": 4})())
        >>> print(gaussian_random_variable({"a": 3, "b": {"c": 4}})())
        >>> print(gaussian_random_variable({"a": 3, "b": {"c": 4, "d":[3,3]}})())
        >>> print(gaussian_random_variable(3)(5))
        >>> print(gaussian_random_variable((3, 4))(5))
        >>> print(gaussian_random_variable({"a": 3, "b": 4})(5))
        >>> print(gaussian_random_variable({"a": 3, "b": {"c": 4}})(5))
        >>> print(gaussian_random_variable(3)((2, 2)))
        >>> print(gaussian_random_variable((3, 4))((2, 2)))
        >>> print(gaussian_random_variable({"a": 3, "b": 4})((2, 2)))
        >>> print(gaussian_random_variable({"a": 3, "b": {"c": 4}})((2, 2)))
        >>> print(gaussian_random_variable({"a": 3, "b": {"c": 4, "d":[3,3]}})((2, 2)))
    """

    def generate_batch_tensor(
        data_size: Union[torch.Tensor, torch.Size, int, Tuple[int], List[int]],
        device: torch.device,
        batch_size: Union[torch.Tensor, torch.Size, int, Tuple[int], List[int]] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Generate a random Gaussian tensor according to the given batch size.
        """
        if isinstance(data_size, torch.Tensor) and len(data_size.shape) == 0:
            data_size = data_size.unsqueeze(0)

        if batch_size is None:
            if isinstance(data_size, int):
                return torch.randn(size=(data_size,), device=device)
            elif (
                isinstance(data_size, Tuple)
                or isinstance(data_size, List)
                or isinstance(data_size, torch.Size)
            ):
                return torch.randn(size=data_size, device=device)
            elif isinstance(data_size, torch.Tensor):
                return torch.randn(size=torch.Size(data_size), device=device)
            else:
                raise ValueError(f"Invalid data size: {data_size}")
        elif isinstance(batch_size, int):
            if isinstance(data_size, int):
                return torch.randn(size=(batch_size, data_size), device=device)
            elif (
                isinstance(data_size, Tuple)
                or isinstance(data_size, List)
                or isinstance(data_size, torch.Size)
            ):
                return torch.randn(size=(batch_size, *data_size), device=device)
            elif isinstance(data_size, torch.Tensor):
                return torch.randn(
                    size=(batch_size, *torch.Size(data_size)), device=device
                )
            else:
                raise ValueError(f"Invalid data size: {data_size}")
        elif (
            isinstance(batch_size, Tuple)
            or isinstance(batch_size, List)
            or isinstance(batch_size, torch.Size)
        ):
            if isinstance(data_size, int):
                return torch.randn(size=(*batch_size, data_size), device=device)
            elif (
                isinstance(data_size, Tuple)
                or isinstance(data_size, List)
                or isinstance(data_size, torch.Size)
            ):
                return torch.randn(size=(*batch_size, *data_size), device=device)
            elif isinstance(data_size, torch.Tensor):
                return torch.randn(
                    size=(*batch_size, *torch.Size(data_size)), device=device
                )
            else:
                raise ValueError(f"Invalid data size: {data_size}")
        elif isinstance(batch_size, torch.Tensor):
            if len(batch_size.shape) == 0:
                batch_size = torch.Size(batch_size.unsqueeze(0))
            if isinstance(data_size, int):
                return torch.randn(size=(*batch_size, data_size), device=device)
            elif (
                isinstance(data_size, Tuple)
                or isinstance(data_size, List)
                or isinstance(data_size, torch.Size)
            ):
                return torch.randn(size=(*batch_size, *data_size), device=device)
            elif isinstance(data_size, torch.Tensor):
                return torch.randn(
                    size=(*batch_size, *torch.Size(data_size)), device=device
                )
            else:
                raise ValueError(f"Invalid data size: {data_size}")
        else:
            raise ValueError(f"Invalid batch size: {batch_size}")

    if isinstance(data_size, Dict):

        def generate_dict(
            data_size: Dict[Any, Any], device: torch.device
        ) -> Dict[Any, Any]:
            """
            Overview:
                Generate a dict of random Gaussian tensor generators according to the given data size if it is a dictionary.
            Arguments:
                data_size (:obj:`Dict[Any, Any]`): The data size.
                device (:obj:`torch.device`): The device.
            """
            data_dict = {}
            for k, v in data_size.items():
                if (
                    isinstance(v, int)
                    or isinstance(v, Tuple)
                    or isinstance(v, List)
                    or isinstance(v, torch.Size)
                    or isinstance(v, torch.Tensor)
                ):
                    data_dict[k] = lambda batch_size=None, v=deepcopy(
                        v
                    ): generate_batch_tensor(v, device, batch_size)
                elif isinstance(v, Dict):
                    data_dict[k] = generate_dict(deepcopy(v), device=device)
                else:
                    raise ValueError(f"Invalid data size: {v}")
            return data_dict

        data_dict = generate_dict(data_size, device)

        def generate_data_from_dict(
            generator_dict: Dict[str, Any],
            device: torch.device,
            batch_size: Union[
                torch.Tensor, torch.Size, int, Tuple[int], List[int]
            ] = None,
            use_tree_tensor: bool = False,
        ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
            """
            Overview:
                Generate a random Gaussian tensor according to the given batch size.
            Arguments:
                batch_size (:obj:`Union[torch.Size, int, Tuple[int], List[int]]`): The batch size.
                generator_dict (:obj:`Dict[str, Any]`): The generator dictionary.
                device (:obj:`torch.device`): The device.
            """

            if isinstance(batch_size, torch.Tensor) and len(batch_size.shape) == 0:
                batch_size = batch_size.unsqueeze(0)
            if use_tree_tensor:
                generated_data = treetensor.torch.Tensor({}, device=device)
            else:
                generated_data = TensorDict(
                    {},
                    batch_size=batch_size if batch_size is not None else {},
                    device=device,
                )
            for k, v in generator_dict.items():
                if isinstance(v, Callable):
                    generated_data[k] = v(batch_size)
                elif isinstance(v, Dict):
                    generated_data[k] = generate_data_from_dict(
                        v,
                        device=device,
                        batch_size=batch_size,
                        use_tree_tensor=use_tree_tensor,
                    )
                else:
                    raise ValueError(f"Invalid generator: {v}")
            return generated_data

        return lambda batch_size=None, use_tree_tensor=use_tree_tensor: generate_data_from_dict(
            data_dict,
            device=device,
            batch_size=batch_size,
            use_tree_tensor=use_tree_tensor,
        )
    elif (
        isinstance(data_size, int)
        or isinstance(data_size, Tuple)
        or isinstance(data_size, List)
        or isinstance(data_size, torch.Size)
        or isinstance(data_size, torch.Tensor)
    ):
        return lambda batch_size=None: generate_batch_tensor(
            data_size, device, batch_size
        )
    else:
        raise ValueError(f"Invalid data size: {data_size}")


if __name__ == "__main__":
    print(gaussian_random_variable(3)(5))
    print(gaussian_random_variable((3, 4))(5))
    print(gaussian_random_variable({"a": 3, "b": 4})(5))
    print(gaussian_random_variable({"a": 3, "b": {"c": 4}})(5))

    print(gaussian_random_variable(3)((2, 2)))
    print(gaussian_random_variable((3, 4))((2, 2)))
    print(gaussian_random_variable({"a": 3, "b": 4})((2, 2)))
    print(gaussian_random_variable({"a": 3, "b": {"c": 4}})((2, 2)))
    print(gaussian_random_variable({"a": 3, "b": {"c": 4, "d": [3, 3]}})((2, 2)))

    print(gaussian_random_variable(3)())
    print(gaussian_random_variable((3, 4))())
    print(gaussian_random_variable({"a": 3, "b": 4})())
    print(gaussian_random_variable({"a": 3, "b": {"c": 4}})())
    print(gaussian_random_variable({"a": 3, "b": {"c": 4, "d": [3, 3]}})())
