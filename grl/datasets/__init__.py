from .d4rl import D4RLDataset
from .qgpo import QGPOCustomizedDataset, QGPOD4RLDataset, QGPODataset, QGPOOnlineDataset
from .gp import (
    GPDataset,
    GPD4RLDataset,
    GPOnlineDataset,
    GPD4RLOnlineDataset,
    GPCustomizedDataset,
)
from .minari_dataset import MinariDataset

DATASETS = {
    "QGPOD4RLDataset".lower(): QGPOD4RLDataset,
    "QGPODataset".lower(): QGPODataset,
    "D4RLDataset".lower(): D4RLDataset,
    "QGPOOnlineDataset".lower(): QGPOOnlineDataset,
    "QGPOCustomizedDataset".lower(): QGPOCustomizedDataset,
    "MinariDataset".lower(): MinariDataset,
    "GPDataset".lower(): GPDataset,
    "GPD4RLDataset".lower(): GPD4RLDataset,
    "GPOnlineDataset".lower(): GPOnlineDataset,
    "GPD4RLOnlineDataset".lower(): GPD4RLOnlineDataset,
    "GPCustomizedDataset".lower(): GPCustomizedDataset,
}


def get_dataset(type: str):
    if type.lower() not in DATASETS:
        raise KeyError(f"Invalid dataset type: {type}")
    return DATASETS[type.lower()]


def create_dataset(config, **kwargs):
    return get_dataset(config.type)(**config.args, **kwargs)
