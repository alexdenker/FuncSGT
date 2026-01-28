import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class PriorDataset(Dataset):
    def __init__(self, n_samples=10000, n_points=100, start_seed=None):
        """
        Args:
            n_samples: Number of samples in the dataset
            n_points: Number of points for each function (x resolution)
            start_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.x = torch.linspace(0, 1, n_points)

        self.start_seed = (
            start_seed if start_seed is not None else np.random.randint(0, 10000)
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Combinations of sine waves
        """
        rng = np.random.RandomState(self.start_seed + idx)
        # Combination of sine waves with different frequencies
        n_terms = rng.randint(1, 8)
        u = torch.zeros_like(self.x)
        for i in range(n_terms):
            freq = rng.rand() * 10 + 1
            phase = rng.rand() * 2 * np.pi
            amplitude = rng.rand() * 0.7 + 0.3
            u += amplitude * torch.sin(freq * np.pi * self.x + phase)
        u = u  # / (u.abs().max() + 1e-8)  # Normalize

        return u.unsqueeze(0)  # return shape (1, n_points)


# Example usage and visualization
if __name__ == "__main__":
    dataset = PriorDataset(n_samples=16, n_points=200, start_seed=0)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        u = dataset[i]
        ax.plot(dataset.x.cpu().numpy(), u.squeeze().cpu().numpy(), linewidth=1.5)
        ax.set_title(f"Sample {i+1}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 2])

    plt.suptitle("Diverse Prior Functions", fontsize=14)
    plt.tight_layout()
    plt.savefig("prior_samples.png", dpi=150)
    plt.show()
