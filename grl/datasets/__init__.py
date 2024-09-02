from .d4rl import D4RLDataset
from .qgpo import (
    QGPODataset,
    QGPOD4RLDataset,
    QGPOOnlineDataset,
    QGPOCustomizedDataset,
    QGPOTensorDictDataset,
    QGPOD4RLTensorDictDataset,
    QGPOCustomizedTensorDictDataset,
    QGPODMcontrolTensorDictDataset,
)
from .gp import (
    GPDataset,
    GPD4RLDataset,
    GPOnlineDataset,
    GPD4RLOnlineDataset,
    GPCustomizedDataset,
    GPTensorDictDataset,
    GPD4RLTensorDictDataset,
    GPCustomizedTensorDictDataset,
    GPDMcontrolTensorDictDataset,

)
from .minari_dataset import MinariDataset

DATASETS = {
    "QGPOD4RLDataset".lower(): QGPOD4RLDataset,
    "QGPODataset".lower(): QGPODataset,
    "D4RLDataset".lower(): D4RLDataset,
    "QGPOOnlineDataset".lower(): QGPOOnlineDataset,
    "QGPOCustomizedDataset".lower(): QGPOCustomizedDataset,
    "QGPOTensorDictDataset".lower(): QGPOTensorDictDataset,
    "QGPOD4RLTensorDictDataset".lower(): QGPOD4RLTensorDictDataset,
    "QGPOCustomizedTensorDictDataset".lower(): QGPOCustomizedTensorDictDataset,
    "MinariDataset".lower(): MinariDataset,
    "GPDataset".lower(): GPDataset,
    "GPD4RLDataset".lower(): GPD4RLDataset,
    "GPOnlineDataset".lower(): GPOnlineDataset,
    "GPD4RLOnlineDataset".lower(): GPD4RLOnlineDataset,
    "GPCustomizedDataset".lower(): GPCustomizedDataset,
    "GPTensorDictDataset".lower(): GPTensorDictDataset,
    "GPD4RLTensorDictDataset".lower(): GPD4RLTensorDictDataset,
    "GPCustomizedTensorDictDataset".lower(): GPCustomizedTensorDictDataset,
    "GPDMcontrolTensorDictDataset".lower():GPDMcontrolTensorDictDataset,
    "QGPODMcontrolTensorDictDataset".lower():QGPODMcontrolTensorDictDataset,
}


def get_dataset(type: str):
    if type.lower() not in DATASETS:
        raise KeyError(f"Invalid dataset type: {type}")
    return DATASETS[type.lower()]


def create_dataset(config, **kwargs):
    return get_dataset(config.type)(**config.args, **kwargs)
