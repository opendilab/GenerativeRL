import unittest
import numpy as np
import torch
from grl.agents import obs_transform, action_transform

# Assume obs_transform and action_transform are defined in the same module or imported properly here.


class TestTransforms(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_obs_transform_numpy(self):
        obs = np.array([1, 2, 3], dtype=np.float32)
        transformed = obs_transform(obs, self.device)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertTrue(transformed.is_floating_point())
        self.assertEqual(transformed.device, self.device)
        np.testing.assert_array_equal(transformed.cpu().numpy(), obs)

    def test_obs_transform_dict(self):
        obs = {
            "a": np.array([1, 2, 3], dtype=np.float32),
            "b": np.array([4, 5, 6], dtype=np.float32),
        }
        transformed = obs_transform(obs, self.device)
        self.assertIsInstance(transformed, dict)
        for k, v in transformed.items():
            self.assertIsInstance(v, torch.Tensor)
            self.assertTrue(v.is_floating_point())
            self.assertEqual(v.device, self.device)
            np.testing.assert_array_equal(v.cpu().numpy(), obs[k])

    def test_obs_transform_tensor(self):
        obs = torch.tensor([1, 2, 3], dtype=torch.float32)
        transformed = obs_transform(obs, self.device)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertTrue(transformed.is_floating_point())
        self.assertEqual(transformed.device, self.device)
        self.assertTrue(torch.equal(transformed.cpu(), obs))

    def test_obs_transform_invalid(self):
        obs = [1, 2, 3]
        with self.assertRaises(ValueError):
            obs_transform(obs, self.device)

    def test_action_transform_dict(self):
        action = {
            "a": torch.tensor([1, 2, 3], dtype=torch.float32),
            "b": torch.tensor([4, 5, 6], dtype=torch.float32),
        }
        transformed = action_transform(action, return_as_torch_tensor=True)
        self.assertIsInstance(transformed, dict)
        for k, v in transformed.items():
            self.assertIsInstance(v, torch.Tensor)
            self.assertFalse(v.is_cuda)
            self.assertTrue(torch.equal(v, action[k].cpu()))

        transformed = action_transform(action, return_as_torch_tensor=False)
        self.assertIsInstance(transformed, dict)
        for k, v in transformed.items():
            self.assertIsInstance(v, np.ndarray)
            np.testing.assert_array_equal(v, action[k].cpu().numpy())

    def test_action_transform_tensor(self):
        action = torch.tensor([1, 2, 3], dtype=torch.float32).to(self.device)
        transformed = action_transform(action, return_as_torch_tensor=True)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertFalse(transformed.is_cuda)
        self.assertTrue(torch.equal(transformed, action.cpu()))

        transformed = action_transform(action, return_as_torch_tensor=False)
        self.assertIsInstance(transformed, np.ndarray)
        np.testing.assert_array_equal(transformed, action.cpu().numpy())

    def test_action_transform_numpy(self):
        action = np.array([1, 2, 3], dtype=np.float32)
        transformed = action_transform(action)
        self.assertIsInstance(transformed, np.ndarray)
        np.testing.assert_array_equal(transformed, action)

    def test_action_transform_invalid(self):
        action = [1, 2, 3]
        with self.assertRaises(ValueError):
            action_transform(action)


if __name__ == "__main__":
    unittest.main()
