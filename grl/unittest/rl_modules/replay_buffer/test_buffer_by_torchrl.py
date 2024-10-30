import unittest
import os
from easydict import EasyDict
from unittest.mock import MagicMock
import tempfile
from grl.rl_modules.replay_buffer.buffer_by_torchrl import (
    GeneralListBuffer,
    TensorDictBuffer,
)
from tensordict import TensorDict
import torch


class TestGeneralListBuffer(unittest.TestCase):

    def setUp(self):
        config = EasyDict(size=10, batch_size=2)
        self.buffer = GeneralListBuffer(config)

    def test_add_and_length(self):
        data = [{"state": 1}, {"state": 2}]
        self.buffer.add(data)
        self.assertEqual(len(self.buffer), 2)

    def test_sample(self):
        data = [{"state": 1}, {"state": 2}]
        self.buffer.add(data)
        sample = self.buffer.sample(batch_size=1)
        self.assertIn(sample[0], data)

    def test_get_item(self):
        data = [{"state": 1}, {"state": 2}]
        self.buffer.add(data)
        self.assertEqual(self.buffer[0], data[0])


class TestTensorDictBuffer(unittest.TestCase):

    def setUp(self):
        config = EasyDict(size=10, batch_size=2)
        self.buffer = TensorDictBuffer(config)

    def test_add_and_length(self):
        data = TensorDict(
            {"state": torch.tensor([[1]]), "action": torch.tensor([[0]])},
            batch_size=[1],
        )
        self.buffer.add(data)
        self.assertEqual(len(self.buffer), 1)

    def test_sample(self):
        data = TensorDict(
            {"state": torch.tensor([[1]]), "action": torch.tensor([[0]])},
            batch_size=[1],
        )
        self.buffer.add(data)
        # TODO: temporarily remove the test for compatibility on GitHub Actions
        # sample = self.buffer.sample(batch_size=1)
        # self.assertTrue(isinstance(sample, TensorDict))

    def test_get_item(self):
        data = TensorDict(
            {"state": torch.tensor([[1]]), "action": torch.tensor([[0]])},
            batch_size=[1],
        )
        self.buffer.add(data)
        item = self.buffer[0]
        self.assertTrue(torch.equal(item["state"], torch.tensor([1])))

    def test_save_without_path(self):
        with self.assertRaises(ValueError):
            self.buffer.save()

    def test_load_without_path(self):
        with self.assertRaises(ValueError):
            self.buffer.load()

    def test_save_and_load_with_path(self):
        data = TensorDict(
            {"state": torch.tensor([[1]]), "action": torch.tensor([[0]])},
            batch_size=[1],
        )
        self.buffer.add(data)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "buffer.pkl")
            self.buffer.save(path)
            buffer_2 = TensorDictBuffer(EasyDict(size=10, batch_size=2))
            buffer_2.load(path)
            self.assertEqual(len(buffer_2), 1)
            item = buffer_2[0]
            self.assertTrue(torch.equal(item["state"], torch.tensor([1])))
