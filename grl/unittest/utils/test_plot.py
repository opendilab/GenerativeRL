import unittest
import os
import numpy as np
from grl.utils.plot import plot_distribution 

class TestPlotDistribution(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the test environment. This runs before each test.
        """
        # Sample data for testing
        self.B = 1000  # Number of samples
        self.N = 4     # Number of features
        self.data = np.random.randn(self.B, self.N)  # Random data for demonstration
        self.save_path = "test_distribution_plot.png"  # Path to save test plot

    def tearDown(self):
        """
        Clean up after the test. This runs after each test.
        """
        # Remove the plot file after the test if it was created
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_plot_creation(self):
        """
        Test if the plot is created and saved to the specified path.
        """
        # Call the plot_distribution function
        plot_distribution(self.data, self.save_path)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.save_path), "The plot file was not created.")

        # Verify the file is not empty
        self.assertGreater(os.path.getsize(self.save_path), 0, "The plot file is empty.")

    def test_plot_size(self):
        """
        Test if the plot can be saved with a specified size and DPI.
        """
        size = (8, 8)
        dpi = 300

        # Call the plot_distribution function with a custom size and DPI
        plot_distribution(self.data, self.save_path, size=size, dpi=dpi)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.save_path), "The plot file was not created.")

        # Verify the file is not empty
        self.assertGreater(os.path.getsize(self.save_path), 0, "The plot file is empty.")
