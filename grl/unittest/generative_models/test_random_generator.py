# Test grl/generative_models/random_generator.py

import unittest
from typing import Dict, Tuple

import torch
from tensordict import TensorDict

from grl.generative_models.random_generator import gaussian_random_variable


class TestGaussianRandomVariable(unittest.TestCase):
    def test_scalar_output(self):
        generator = gaussian_random_variable(3)
        tensor = generator()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3]))

    def test_tensor_output(self):
        generator = gaussian_random_variable((3, 4))
        tensor = generator()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 4]))

    def test_tensordict_output(self):
        data_size = {"a": 3, "b": 4}
        generator = gaussian_random_variable(data_size)
        tensor_dict = generator()
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([3]))
        self.assertEqual(tensor_dict["b"].shape, torch.Size([4]))

    def test_nested_tensordict_output(self):
        data_size = {"a": 3, "b": {"c": 4, "d": [3, 3]}}
        generator = gaussian_random_variable(data_size)
        tensor_dict = generator()
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([3, 3]))

    def test_scalar_batch_output(self):
        generator = gaussian_random_variable(3)
        tensor = generator(batch_size=5)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3]))

        tensor = generator(batch_size=(5,))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3]))

        tensor = generator(batch_size=[5])
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3]))

        tensor = generator(batch_size=torch.tensor(5))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3]))

    def test_tensor_batch_output(self):
        generator = gaussian_random_variable((3, 4))
        tensor = generator(batch_size=5)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3, 4]))

        tensor = generator(batch_size=(5,))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3, 4]))

        tensor = generator(batch_size=[5])
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3, 4]))

        tensor = generator(batch_size=torch.tensor(5))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([5, 3, 4]))

    def test_tensordict_batch_output(self):
        data_size = {"a": 3, "b": 4}
        generator = gaussian_random_variable(data_size)
        tensor_dict = generator(batch_size=5)
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertEqual(tensor_dict["b"].shape, torch.Size([5, 4]))

        tensor_dict = generator(batch_size=(5,))
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertEqual(tensor_dict["b"].shape, torch.Size([5, 4]))

        tensor_dict = generator(batch_size=[5])
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertEqual(tensor_dict["b"].shape, torch.Size([5, 4]))

        tensor_dict = generator(batch_size=torch.tensor(5))
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertEqual(tensor_dict["b"].shape, torch.Size([5, 4]))

    def test_nested_tensordict_batch_output(self):
        data_size = {"a": 3, "b": {"c": 4, "d": [3, 3]}}
        generator = gaussian_random_variable(data_size)
        tensor_dict = generator(batch_size=5)
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([5, 4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([5, 3, 3]))

        tensor_dict = generator(batch_size=(5,))
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([5, 4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([5, 3, 3]))

        tensor_dict = generator(batch_size=[5])
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([5, 4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([5, 3, 3]))

        tensor_dict = generator(batch_size=torch.tensor(5))
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([5, 4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([5, 3, 3]))

        tensor_dict = generator(batch_size=torch.tensor([5, 2]))
        self.assertIsInstance(tensor_dict, TensorDict)
        self.assertSetEqual(set(tensor_dict.keys()), set(data_size.keys()))
        self.assertEqual(tensor_dict["a"].shape, torch.Size([5, 2, 3]))
        self.assertIsInstance(tensor_dict["b"], TensorDict)
        self.assertEqual(tensor_dict["b"]["c"].shape, torch.Size([5, 2, 4]))
        self.assertEqual(tensor_dict["b"]["d"].shape, torch.Size([5, 2, 3, 3]))


if __name__ == "__main__":
    test_class = TestGaussianRandomVariable()
    test_class.test_scalar_output()
    test_class.test_tensor_output()
    test_class.test_tensordict_output()
    test_class.test_nested_tensordict_output()
    test_class.test_scalar_batch_output()
    test_class.test_tensor_batch_output()
    test_class.test_tensordict_batch_output()
    test_class.test_nested_tensordict_batch_output()
