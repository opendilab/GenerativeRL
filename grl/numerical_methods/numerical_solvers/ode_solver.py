from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import treetensor
from tensordict import TensorDict
from torch import nn
from torchdiffeq import odeint as torchdiffeq_odeint
from torchdiffeq import odeint_adjoint as torchdiffeq_odeint_adjoint
from torchdyn.core import NeuralODE
from torchdyn.numerics import Euler
from torchdyn.numerics import odeint as torchdyn_odeint


class ODESolver:
    """
    Overview:
        The ODE solver class.
    Interfaces:
        ``__init__``, ``integrate``
    """

    def __init__(
        self,
        ode_solver="euler",
        dt=0.01,
        atol=1e-5,
        rtol=1e-5,
        library="torchdyn",
        **kwargs,
    ):
        """
        Overview:
            Initialize the ODE solver using torchdiffeq or torchdyn library.
        Arguments:
            ode_solver (:obj:`str`): The ODE solver to use.
            dt (:obj:`float`): The time step.
            atol (:obj:`float`): The absolute tolerance.
            rtol (:obj:`float`): The relative tolerance.
            library (:obj:`str`): The library to use for the ODE solver. Currently, it supports 'torchdiffeq' and 'torchdyn'.
            **kwargs: Additional arguments for the ODE solver.
        """
        self.ode_solver = ode_solver
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.library = library

    def integrate(
        self,
        drift: Union[nn.Module, Callable],
        x0: Union[torch.Tensor, TensorDict],
        t_span: torch.Tensor,
        **kwargs,
    ):
        """
        Overview:
            Integrate the ODE.
        Arguments:
            drift (:obj:`Union[nn.Module, Callable]`): The drift term of the ODE.
            x0 (:obj:`Union[torch.Tensor, TensorDict]`): The input initial state.
            t_span (:obj:`torch.Tensor`): The time at which to evaluate the ODE. The first element is the initial time, and the last element is the final time. For example, t = torch.tensor([0.0, 1.0]).
        Returns:
            trajectory (:obj:`Union[torch.Tensor, TensorDict]`): The output trajectory of the ODE, which has the same data type as x0 and the shape of (len(t_span), *x0.shape).
        """

        self.nfe = 0
        if self.library == "torchdiffeq":
            return self.odeint_by_torchdiffeq(drift, x0, t_span)
        elif self.library == "torchdiffeq_adjoint":
            return self.odeint_by_torchdiffeq_adjoint(drift, x0, t_span, **kwargs)
        elif self.library == "torchdyn":
            return self.odeint_by_torchdyn(drift, x0, t_span)
        elif self.library == "torchdyn_NeuralODE":
            return self.odeint_by_torchdyn_NeuralODE(drift, x0, t_span)
        else:
            raise ValueError(f"library {self.library} is not supported")

    def odeint_by_torchdiffeq(self, drift, x0, t_span, **kwargs):

        if isinstance(x0, torch.Tensor):

            def forward_ode_drift_by_torchdiffeq(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x.shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint(
                func=forward_ode_drift_by_torchdiffeq,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        elif isinstance(x0, Tuple):

            def forward_ode_drift_by_torchdiffeq(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x[0].shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint(
                func=forward_ode_drift_by_torchdiffeq,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        else:
            raise ValueError(f"Unsupported data type for x0: {type(x0)}")

    def odeint_by_torchdiffeq_adjoint(self, drift, x0, t_span, **kwargs):

        if isinstance(x0, torch.Tensor):

            def forward_ode_drift_by_torchdiffeq_adjoint(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x.shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint_adjoint(
                func=forward_ode_drift_by_torchdiffeq_adjoint,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

        elif isinstance(x0, Tuple):

            def forward_ode_drift_by_torchdiffeq_adjoint(t, x):
                self.nfe += 1
                # broadcasting t to match the batch size of x
                t = t.repeat(x[0].shape[0])
                return drift(t, x)

            trajectory = torchdiffeq_odeint_adjoint(
                func=forward_ode_drift_by_torchdiffeq_adjoint,
                y0=x0,
                t=t_span,
                method=self.ode_solver,
                atol=self.atol,
                rtol=self.rtol,
                **self.kwargs,
                **kwargs,
            )
            return trajectory

    def odeint_by_torchdyn(self, drift, x0, t_span):

        def forward_ode_drift_by_torchdyn(t, x):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            return drift(t, x)

        t_eval, trajectory = torchdyn_odeint(
            f=forward_ode_drift_by_torchdyn,
            x=x0,
            t_span=t_span,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            **self.kwargs,
        )
        return trajectory

    def odeint_by_torchdyn_NeuralODE(self, drift, x0, t_span):

        def forward_ode_drift_by_torchdyn_NeuralODE(t, x, args):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            return drift(t, x)

        neural_ode = NeuralODE(
            vector_field=forward_ode_drift_by_torchdyn_NeuralODE,
            sensitivity="adjoint",
            solver_adjoint="dopri5",
            atol_adjoint=1e-5,
            rtol_adjoint=1e-5,
            solver=self.ode_solver,
            # atol=self.atol,
            # rtol=self.rtol,
            return_t_eval=False,
            **self.kwargs,
        )
        trajectory = neural_ode(x0, t_span)
        return trajectory


class DictTensorConverter(nn.Module):

    def __init__(self, dict_type: object = None) -> None:
        """
        Overview:
            Initialize the DictTensorConverter module.
        Arguments:
            dict_type (:obj:`object`): The type of the dictionary to be used, which can be (:obj:`dict`), (:obj:`TensorDict`) or (:obj:`treetensor.torch.Tensor`).
        """
        super().__init__()
        self.dict_type = dict_type if dict_type is not None else dict
        assert self.dict_type in [dict, TensorDict, treetensor.torch.Tensor]

    def dict_to_tensor(
        self,
        input_dict: Dict[str, torch.Tensor],
        batch_size: Union[int, torch.Size, torch.Tensor, List, Tuple],
    ) -> torch.Tensor:
        """
        Overview:
            Convert a dictionary of PyTorch tensors into a single PyTorch tensor.

        Arguments:
            input_dict (:obj:`dict`): A dictionary where the values are PyTorch tensors.
            batch_size (:obj:`Union[int, torch.Size, torch.Tensor, List, Tuple]`): The batch size or shape of the batch dimensions.

        Returns:
            torch.Tensor (:obj:`torch.Tensor`): A single PyTorch tensor containing the concatenated values from the input dictionary.
        """
        tensor_list = []
        keys = list(input_dict.keys())
        keys.sort()
        for key in keys:
            value = input_dict[key]
            if isinstance(batch_size, int):
                tensor_list.append(value.reshape(batch_size, -1))
            elif isinstance(batch_size, torch.Size):
                tensor_list.append(value.reshape(batch_size, -1))
            elif isinstance(batch_size, torch.Tensor):
                if len(batch_size.shape) == 0:
                    tensor_list.append(value.reshape(batch_size.item(), -1))
                else:
                    tensor_list.append(value.reshape((*(batch_size.tolist()), -1)))
            elif isinstance(batch_size, (list, tuple)):
                tensor_list.append(value.reshape((*batch_size, -1)))
            else:
                raise TypeError(f"Unsupported type for batch_size: {type(batch_size)}")

        combined_tensor = torch.cat(tensor_list, dim=-1)
        return combined_tensor

    def tensor_to_dict(
        self,
        input_tensor: torch.Tensor,
        x_size: Dict[str, Union[int, torch.Size, torch.Tensor, List, Tuple]],
    ) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Convert a single PyTorch tensor into a dictionary of PyTorch tensors.

        Arguments:
            input_tensor (:obj:`torch.Tensor`): A PyTorch tensor of shape (*batch_size, sum_dimensions).
            x_size (:obj:`dict`): A dictionary where the keys are the names of the tensors, and the values are the dimensions.

        Returns:
            dict (:obj:`dict`): A dictionary where the keys are the names of the tensors and the values are PyTorch tensors with the specified dimensions.
        """

        output_dict = {}
        start_idx = 0
        keys = list(x_size.keys())
        keys.sort()
        for key in keys:
            dimensions = x_size[key]
            if isinstance(dimensions, int):
                assert dimensions == torch.prod(torch.tensor(input_tensor.shape[:-1]))
                end_idx = start_idx + 1
            elif isinstance(dimensions, torch.Size):
                data_size = dimensions[len(input_tensor.shape[:-1]) :]
                end_idx = (
                    start_idx
                    + torch.prod(torch.tensor(data_size, dtype=torch.int)).item()
                )
            elif isinstance(dimensions, torch.Tensor):
                data_size = dimensions[len(input_tensor.shape[:-1]) :]
                end_idx = start_idx + torch.prod(data_size, dtype=torch.int).item()
            elif isinstance(dimensions, (list, tuple)):
                data_size = dimensions[len(input_tensor.shape[:-1]) :]
                end_idx = (
                    start_idx
                    + torch.prod(torch.tensor(data_size, dtype=torch.int)).item()
                )
            else:
                raise TypeError(f"Unsupported type for dimensions: {type(dimensions)}")
            output_dict[key] = input_tensor[..., start_idx:end_idx].reshape(dimensions)
            start_idx = end_idx

        if self.dict_type == dict:
            return output_dict
        elif self.dict_type == TensorDict:
            return TensorDict(output_dict, batch_size=x_size.batch_size)
        elif self.dict_type == treetensor.torch.Tensor:
            output = treetensor.torch.tensor({}, device=input_tensor.device)
            for key in output_dict.keys():
                output[key] = output_dict[key]
            return output
        else:
            raise ValueError(f"Unsupported dictionary type: {self.dict_type}")


class DictTensorODESolver:
    """
    Overview:
        The ODE solver class for dict type input.
    Interfaces:
        ``__init__``, ``integrate``
    """

    def __init__(
        self,
        ode_solver="euler",
        dt=0.01,
        atol=1e-5,
        rtol=1e-5,
        library="torchdyn",
        dict_type: str = None,
        **kwargs,
    ):
        """
        Overview:
            Initialize the ODE solver using torchdiffeq or torchdyn library.
        Arguments:
            ode_solver (:obj:`str`): The ODE solver to use.
            dt (:obj:`float`): The time step.
            atol (:obj:`float`): The absolute tolerance.
            rtol (:obj:`float`): The relative tolerance.
            library (:obj:`str`): The library to use for the ODE solver. Currently, it supports 'torchdyn' and 'torchdiffeq'.
            **kwargs: Additional arguments for the ODE solver.
        """
        self.ode_solver = ode_solver
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.kwargs = kwargs
        self.library = library

        self.dict_type = dict_type
        if dict_type is None or dict_type == "dict":
            self.dict_tensor_converter = DictTensorConverter(dict_type=dict)
        elif dict_type == "TensorDict":
            self.dict_tensor_converter = DictTensorConverter(dict_type=TensorDict)
        elif dict_type == "treetensor":
            self.dict_tensor_converter = DictTensorConverter(
                dict_type=treetensor.torch.Tensor
            )
        else:
            raise ValueError(f"Unsupported dict_type: {dict_type}")

    def integrate(
        self,
        drift: Union[nn.Module, Callable],
        x0: Union[dict, TensorDict, treetensor.torch.Tensor],
        t_span: torch.Tensor,
        batch_size: Union[int, torch.Size, torch.Tensor, List, Tuple],
        x_size: Dict[str, Union[int, torch.Size, torch.Tensor, List, Tuple]],
    ):
        """
        Overview:
            Integrate the ODE.
        Arguments:
            drift (:obj:`Union[nn.Module, Callable]`): The drift term of the ODE.
            x0 (:obj:`Union[dict, TensorDict, treetensor.torch.Tensor]`): The input initial state.
            t_span (:obj:`torch.Tensor`): The time at which to evaluate the ODE. The first element is the initial time, and the last element is the final time. For example, t = torch.tensor([0.0, 1.0]).
        Returns:
            trajectory (:obj:`Union[dict, TensorDict, treetensor.torch.Tensor]`): The output trajectory of the ODE, which has the same data type as x0 and the shape of (len(t_span), *x0.shape).
        """

        self.nfe = 0

        if self.library == "torchdyn":
            return self.odeint_by_torchdyn(drift, x0, t_span, batch_size, x_size)
        elif self.library == "torchdyn_NeuralODE":
            return self.odeint_by_torchdyn_NeuralODE(
                drift, x0, t_span, batch_size, x_size
            )
        else:
            raise ValueError(f"library {self.library} is not supported")

    def odeint_by_torchdyn(self, drift, x0, t_span, batch_size, x_size):

        def forward_ode_drift_by_torchdyn(t, x):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            # x.shape = (*batch_size, *sum_dimensions)
            x = self.dict_tensor_converter.tensor_to_dict(x, x_size=x_size)
            # x is a dictionary of PyTorch tensors
            # x[key].shape = (*x_size[key])
            dict_drift = drift(t, x)
            # dict_drift is a dictionary of PyTorch tensors
            # dict_drift[key].shape = (*x_size[key])
            return self.dict_tensor_converter.dict_to_tensor(
                dict_drift, batch_size=batch_size
            )

        x0 = self.dict_tensor_converter.dict_to_tensor(x0, batch_size=batch_size)
        # x0.shape = (*batch_size, *sum_dimensions)

        t_eval, trajectory = torchdyn_odeint(
            f=forward_ode_drift_by_torchdyn,
            x=x0,
            t_span=t_span,
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            **self.kwargs,
        )

        if self.dict_type == "dict":
            raise NotImplementedError("dict type is not supported")
        elif self.dict_type == "TensorDict":
            raise NotImplementedError("TensorDict type is not supported")
        elif self.dict_type == "treetensor":
            t_span_tensordict = treetensor.torch.tensor({}, device=t_span.device)
            for key in x_size.keys():
                t_span_tensordict[key] = torch.tensor(
                    [t_span.numel()], device=t_span.device
                )
            t_x_size = treetensor.torch.Size(
                treetensor.torch.cat(
                    [
                        t_span_tensordict,
                        treetensor.torch.tensor(x_size, device=t_span.device),
                    ],
                    dim=0,
                )
            )

            trajectory = self.dict_tensor_converter.tensor_to_dict(
                trajectory, x_size=t_x_size
            )

        return trajectory

    def odeint_by_torchdyn_NeuralODE(self, drift, x0, t_span, batch_size, x_size):

        def forward_ode_drift_by_torchdyn_NeuralODE(t, x, args={}):
            self.nfe += 1
            # broadcasting t to match the batch size of x
            t = t.repeat(x.shape[0])
            # x.shape = (*batch_size, *sum_dimensions)
            x = self.dict_tensor_converter.tensor_to_dict(x, x_size=x_size)
            # x is a dictionary of PyTorch tensors
            # x[key].shape = (*x_size[key])
            dict_drift = drift(t, x)
            # dict_drift is a dictionary of PyTorch tensors
            # dict_drift[key].shape = (*x_size[key])
            return self.dict_tensor_converter.dict_to_tensor(
                dict_drift, batch_size=batch_size
            )

        neural_ode = NeuralODE(
            vector_field=forward_ode_drift_by_torchdyn_NeuralODE,
            sensitivity="adjoint",
            solver=self.ode_solver,
            atol=self.atol,
            rtol=self.rtol,
            return_t_eval=False,
            **self.kwargs,
        )

        x0 = self.dict_tensor_converter.dict_to_tensor(x0, batch_size=batch_size)

        trajectory = neural_ode(x0, t_span)

        if self.dict_type == "dict":
            raise NotImplementedError("dict type is not supported")
        elif self.dict_type == "TensorDict":
            raise NotImplementedError("TensorDict type is not supported")
        elif self.dict_type == "treetensor":
            t_span_tensordict = treetensor.torch.tensor({}, device=t_span.device)
            for key in x_size.keys():
                t_span_tensordict[key] = torch.tensor(
                    [t_span.numel()], device=t_span.device
                )
            t_x_size = treetensor.torch.Size(
                treetensor.torch.cat(
                    [
                        t_span_tensordict,
                        treetensor.torch.tensor(x_size, device=t_span.device),
                    ],
                    dim=0,
                )
            )

            trajectory = self.dict_tensor_converter.tensor_to_dict(
                trajectory, x_size=t_x_size
            )

        return trajectory
