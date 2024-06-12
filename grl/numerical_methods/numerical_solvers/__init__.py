from .dpm_solver import DPMSolver
from .ode_solver import DictTensorODESolver, ODESolver
from .sde_solver import SDESolver


def get_solver(solver_type):
    if solver_type.lower() in SOLVERS:
        return SOLVERS[solver_type.lower()]
    else:
        raise ValueError(f"Solver type {solver_type} not recognized")


SOLVERS = {
    "DPMSolver".lower(): DPMSolver,
    "ODESolver".lower(): ODESolver,
    "DictTensorODESolver".lower(): DictTensorODESolver,
    "SDESolver".lower(): SDESolver,
}
