from easydict import EasyDict
from typing import List, Union
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    SamplerWithoutReplacement,
    RandomSampler,
)
from torchrl.data import (
    TensorStorage,
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)


class GeneralListBuffer:
    """
    Overview:
        GeneralListBuffer is a general buffer for storing list data.
    Interface:
        ``__init__``, ``add``, ``sample``, ``__len__``, ``__getitem__``, ``__setitem__``, ``__delitem__``, ``__iter__``, ``__contains__``, ``__repr__``, ``save``, ``load``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the buffer.
        Arguments:
            config (:obj:`EasyDict`): Config dict, which contains the following keys:
                - size (:obj:`int`): Size of the buffer.
                - batch_size (:obj:`int`, optional): Batch size.
        """
        self.config = config
        self.size = config.size
        self.batch_size = config.get("batch_size", 1)
        self.path = config.get("path", None)

        self.storage = ListStorage(max_size=self.size)
        self.buffer = ReplayBuffer(
            storage=self.storage, batch_size=self.batch_size, collate_fn=lambda x: x
        )

    def add(self, data: List):
        """
        Overview:
            Add data to the buffer.
        Arguments:
            data (:obj:`List`): Data to be added.
        """
        self.buffer.extend(data)

    def sample(self, batch_size: int = None):
        """
        Overview:
            Sample data from the buffer.
        Arguments:
            batch_size (:obj:`int`): Batch size.
        Returns:
            (:obj:`List`): Sampled data.
        """
        return self.buffer.sample(batch_size=batch_size)

    def __len__(self):
        """
        Overview:
            Get the length of the buffer.
        Returns:
            (:obj:`int`): Length of the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, index: int):
        """
        Overview:
            Get item by index.
        Arguments:
            index (:obj:`int`): Index.
        Returns:
            (:obj:`dict`): Item.
        """
        return self.storage.get(index=index)

    def __setitem__(self, index: Union[int, List], data: dict):
        """
        Overview:
            Set item by index.
        Arguments:
            index (:obj:`Union[int, List]`): Index.
            data (:obj:`dict`): Data.
        """
        self.storage.set(cursor=index, data=data)

    def __delitem__(self, index: int):
        """
        Overview:
            Delete item by index.
        Arguments:
            index (:obj:`int`): Index.
        """
        del self.buffer[index]

    def __iter__(self):
        """
        Overview:
            Iterate the buffer.
        Returns:
            (:obj:`iter`): Iterator.
        """
        return iter(self.buffer)

    def __contains__(self, item: dict):
        """
        Overview:
            Check if the item is in the buffer.
        Arguments:
            item (:obj:`dict`): Item.
        """
        return item in self.buffer

    def __repr__(self):
        """
        Overview:
            Get the representation of the buffer.
        Returns:
            (:obj:`str`): Representation of the buffer.
        """
        return repr(self.buffer)

    def save(self, path: str = None):
        raise NotImplementedError("GeneralListBuffer does not support save method.")
        # TODO: Implement save method
        # path = path if path is not None else self.path
        # if path is None:
        #     raise ValueError("Path is not provided.")
        # self.buffer.dump(path)

    def load(self, path: str = None):
        raise NotImplementedError("GeneralListBuffer does not support load method.")
        # TODO: Implement load method
        # path = path if path is not None else self.path
        # if path is None:
        #     raise ValueError("Path is not provided.")
        # self.buffer.load(path)


class TensorDictBuffer:
    """
    Overview:
        TensorDictBuffer is a buffer for storing TensorDict data, which use TensorDictReplayBuffer as the underlying buffer.
    Interface:
        ``__init__``, ``add``, ``sample``, ``__len__``, ``__getitem__``, ``__setitem__``, ``__delitem__``, ``__iter__``, ``__contains__``, ``__repr__``, ``save``, ``load``
    """

    def __init__(self, config: EasyDict, data: TensorDict = None):
        """
        Overview:
            Initialize the buffer.
        Arguments:
            config (:obj:`EasyDict`): Config dict, which contains the following keys:
                - size (:obj:`int`): Size of the buffer.
                - memory_map (:obj:`bool`, optional): Whether to use memory map.
                - replacement (:obj:`bool`, optional): Whether to use replacement.
                - drop_last (:obj:`bool`, optional): Whether to drop the last batch.
                - shuffle (:obj:`bool`, optional): Whether to shuffle the data.
                - prefetch (:obj:`int`, optional): Number of prefetch.
                - pin_memory (:obj:`bool`, optional): Whether to pin memory.
                - batch_size (:obj:`int`, optional): Batch size.
                - path (:obj:`str`, optional): Path to save the buffer.
            data (:obj:`TensorDict`, optional): Data to be stored.
        """
        self.config = config
        self.size = config.size
        self.lazy_init = True if data is None else False
        self.memory_map = config.get("memory_map", False)
        self.replacement = config.get("replacement", False)
        self.drop_last = config.get("drop_last", False)
        self.shuffle = config.get("shuffle", False)
        self.prefetch = config.get("prefetch", 10)
        self.pin_memory = config.get("pin_memory", True)
        self.batch_size = config.get("batch_size", 1)
        self.path = config.get("path", None)

        if self.lazy_init:
            if self.memory_map:
                self.storage = LazyMemmapStorage(
                    max_size=self.size,
                    scratch_dir=config.scratch_dir if "scratch_dir" in config else None,
                )
            else:
                self.storage = LazyTensorStorage(max_size=self.size)
        else:
            self.storage = TensorStorage(storage=data, max_size=self.size)

        if self.replacement:
            self.sampler = SamplerWithoutReplacement(
                drop_last=self.drop_last, shuffle=self.shuffle
            )
        else:
            self.sampler = RandomSampler()

        self.buffer = TensorDictReplayBuffer(
            storage=self.storage,
            batch_size=self.batch_size,
            sampler=self.sampler,
            prefetch=self.prefetch,
            pin_memory=self.pin_memory,
        )

    def add(self, data: Union[TensorDict, dict]):
        """
        Overview:
            Add data to the buffer.
        Arguments:
            data (:obj:`Union[TensorDict, dict]`): Data to be added.
        """
        if isinstance(data, dict):
            data = TensorDict(data)
        self.buffer.extend(data)

    def sample(self, batch_size: int = None):
        """
        Overview:
            Sample data from the buffer.
        Arguments:
            batch_size (:obj:`int`): Batch size.
        """
        return self.buffer.sample(batch_size=batch_size)

    def __len__(self):
        """
        Overview:
            Get the length of the buffer.
        Returns:
            (:obj:`int`): Length of the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, index: int):
        """
        Overview:
            Get item by index.
        Arguments:
            index (:obj:`int`): Index.
        """
        return self.storage.get(index=index)

    def __setitem__(self, index: Union[int, List], data: dict):
        """
        Overview:
            Set item by index.
        Arguments:
            index (:obj:`Union[int, List]`): Index.
            data (:obj:`dict`): Data.
        """
        self.storage.set(cursor=index, data=data)

    def __delitem__(self, index: int):
        """
        Overview:
            Delete item by index.
        Arguments:
            index (:obj:`int`): Index
        """
        del self.buffer[index]

    def __iter__(self):
        """
        Overview:
            Iterate the buffer.
        Returns:
            (:obj:`iter`): Iterator.
        """
        return iter(self.buffer)

    def __contains__(self, item: dict):
        """
        Overview:
            Check if the item is in the buffer.
        Arguments:
            item (:obj:`dict`): Item.
        """
        return item in self.buffer

    def __repr__(self):
        """
        Overview:
            Get the representation of the buffer.
        Returns:
            (:obj:`str`): Representation of the buffer.
        """
        return repr(self.buffer)

    def save(self, path: str = None):
        """
        Overview:
            Save the buffer.
        Arguments:
            path (:obj:`str`, optional): Path to save the buffer.
        """
        path = path if path is not None else self.path
        if path is None:
            raise ValueError("Path is not provided.")
        self.buffer.dump(path)

    def load(self, path: str = None):
        """
        Overview:
            Load the buffer.
        Arguments:
            path (:obj:`str`, optional): Path to load the buffer.
        """
        path = path if path is not None else self.path
        if path is None:
            raise ValueError("Path is not provided.")
        self.buffer.load(path)
