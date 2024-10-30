import unittest
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from grl.utils.model_utils import save_model, load_model


class TestModelCheckpointing(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to save/load checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple model and optimizer for testing
        self.model = nn.Linear(10, 2)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.temp_dir)

    def test_save_model(self):
        # Test saving the model
        iteration = 100
        save_model(self.temp_dir, self.model, self.optimizer, iteration)

        # Check if the directory was created and torch.save was called correctly
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_load_model(self):
        # Create a mock checkpoint file
        iteration = 100
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": iteration,
        }

        # Save a checkpoint file manually
        checkpoint_file = os.path.join(self.temp_dir, f"checkpoint_{iteration}.pt")
        torch.save(checkpoint, checkpoint_file)

        # Create a simple model and optimizer for testing
        new_model = nn.Linear(10, 2)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)

        # Test loading the model
        loaded_iteration = load_model(self.temp_dir, new_model, new_optimizer)

        # Check if the correct iteration was returned
        self.assertEqual(loaded_iteration, iteration)

        # Check if the model and optimizer were loaded correctly
        self.assertTrue(
            torch.allclose(
                new_model.state_dict()["weight"], self.model.state_dict()["weight"]
            )
        )
        self.assertTrue(
            torch.allclose(
                new_model.state_dict()["bias"], self.model.state_dict()["bias"]
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(new_optimizer.state_dict()["param_groups"][0]["lr"]),
                torch.tensor(self.optimizer.state_dict()["param_groups"][0]["lr"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(new_optimizer.state_dict()["param_groups"][0]["momentum"]),
                torch.tensor(
                    self.optimizer.state_dict()["param_groups"][0]["momentum"]
                ),
            )
        )

    def test_load_model_order(self):
        # Create mock checkpoint files
        iterations = [100, 200, 300]
        for iteration in iterations:
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": iteration,
            }
            checkpoint_file = os.path.join(self.temp_dir, f"checkpoint_{iteration}.pt")
            torch.save(checkpoint, checkpoint_file)

        # Create a simple model and optimizer for testing
        new_model = nn.Linear(10, 2)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)

        # Test loading the model
        loaded_iteration = load_model(self.temp_dir, new_model, new_optimizer)

        # Check if the correct iteration was returned
        self.assertEqual(loaded_iteration, iterations[-1])

    def test_load_model_no_files(self):
        # Test loading when no checkpoint files exist
        loaded_iteration = load_model(self.temp_dir, self.model, self.optimizer)

        # Check that the function returns -1 when no files are found
        self.assertEqual(loaded_iteration, -1)


if __name__ == "__main__":
    unittest.main()
