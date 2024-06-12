import math
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.distributions.uniform as uniform
import torch.nn.functional as F


class MonteCarloSampler:
    """
    Overview:
        A class to sample from an unnormalized PDF using Monte Carlo sampling.
    Interface:
        ``__init__``, ``sample``, ``plot_samples``
    """

    def __init__(
        self,
        unnormalized_pdf: Callable,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Overview:
            Initialize the Monte Carlo sampler.
        Arguments:
            - unnormalized_pdf (:obj:`Callable`): The unnormalized PDF function.
            - x_min (:obj:`torch.Tensor`): The minimum value of the range.
            - x_max (:obj:`torch.Tensor`): The maximum value of the range.
        """
        self.unnormalized_pdf = unnormalized_pdf
        self.x_min = x_min
        self.x_max = x_max
        self.device = device
        self.uniform_dist = uniform.Uniform(
            torch.tensor(self.x_min, device=device),
            torch.tensor(self.x_max, device=device),
        )

    def sample(self, num_samples: int):
        """
        Overview:
            Sample from the unnormalized PDF using Monte Carlo sampling.
        """

        # if the number of accepted samples is less than the number of samples, sample more
        samples = torch.tensor([], device=self.device)
        sample_ratio = 1.0
        while len(samples) < num_samples:
            num_to_sample = math.floor((num_samples - len(samples)) * sample_ratio)
            # if num_to_sample is larger than INT_MAX, sample no more than INT_MAX samples
            if num_to_sample > 2**24:
                num_to_sample = 2**24
            samples_ = self._sample(num_to_sample)
            sample_ratio = num_to_sample / samples_.shape[0]
            samples = torch.cat([samples, samples_])

        # randomly drop samples to get the exact number of samples
        samples = samples[:num_samples]
        return samples

    def _sample(self, num_samples: int):

        # Normalize the PDF
        # x = torch.linspace(self.x_min, self.x_max, eval_num)
        # pdf_values = self.unnormalized_pdf(x)
        # normalization_constant = torch.trapz(pdf_values, x)
        # normalized_pdf = self.unnormalized_pdf(x) / normalization_constant

        random_samples = self.uniform_dist.sample((num_samples,))

        # Evaluate PDF values
        pdf_samples = self.unnormalized_pdf(random_samples)

        # Normalize PDF values
        normalized_pdf_samples = pdf_samples / torch.max(pdf_samples)

        # Accept or reject samples
        accepted_samples = random_samples[
            torch.rand(num_samples, device=self.device) < normalized_pdf_samples
        ]

        return accepted_samples

    def plot_samples(self, samples, num_bins=50):
        plt.figure(figsize=(10, 6))
        plt.hist(
            samples.detach().cpu().numpy(),
            bins=num_bins,
            density=True,
            alpha=0.5,
            label="Monte Carlo samples",
        )
        x = torch.linspace(self.x_min, self.x_max, 1000)
        normalized_pdf = self.unnormalized_pdf(x) / torch.trapz(
            self.unnormalized_pdf(x), x
        )
        plt.plot(x, normalized_pdf, color="red", label="Normalized PDF")
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.title("Sampling from an Unnormalized PDF using Monte Carlo")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Define the unnormalized PDF function
    def unnormalized_pdf(x):
        return torch.exp(-0.5 * (x - 0.5) ** 2) + 0.5 * torch.sin(2 * torch.pi * x)

    # Define the range [0, 1]
    x_min = 0.0
    x_max = 1.0

    # Initialize the Monte Carlo sampler
    monte_carlo_sampler = MonteCarloSampler(unnormalized_pdf, x_min, x_max)

    # Sample from the unnormalized PDF
    num_samples = 10000
    samples = monte_carlo_sampler.sample(num_samples)
    assert len(samples) == num_samples

    # Plot the samples
    monte_carlo_sampler.plot_samples(samples)
