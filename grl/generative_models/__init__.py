def get_generative_model(name: str):
    if name.lower() not in GENERATIVE_MODELS:
        raise ValueError("Unknown activation function {}".format(name))
    return GENERATIVE_MODELS[name.lower()]


from .conditional_flow_model import (
    IndependentConditionalFlowModel,
    OptimalTransportConditionalFlowModel,
)
from .diffusion_model import DiffusionModel, EnergyConditionalDiffusionModel
from .variational_autoencoder import VariationalAutoencoder

GENERATIVE_MODELS = {
    "DiffusionModel".lower(): DiffusionModel,
    "EnergyConditionalDiffusionModel".lower(): EnergyConditionalDiffusionModel,
    "VariationalAutoencoder".lower(): VariationalAutoencoder,
    "IndependentConditionalFlowModel".lower(): IndependentConditionalFlowModel,
    "OptimalTransportConditionalFlowModel".lower(): OptimalTransportConditionalFlowModel,
}
