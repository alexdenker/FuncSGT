import torch 
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class PriorDataset(Dataset):
    def __init__(self, n_samples=1000, n_points=100):
        """
        Args:
            n_samples: Number of samples in the dataset
            n_points: Number of points for each function (x resolution)
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.x = torch.linspace(0, 1, n_points)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate random lambda parameter
        lamb = np.random.rand() * 0.3 + 0.2 # in [0.2, 0.5]
        scaling = np.random.rand() * 50 + 75  # in [75, 125]
        overall_scaling = np.random.rand() * 0.2 + 0.9  # in [0.9, 1.1]
        # Generate prior function
        u = overall_scaling * (torch.exp(-scaling*(self.x - lamb)**2) - torch.exp(-scaling*(self.x - (1-lamb))**2))

        return u.unsqueeze(0)  # return shape (1, n_points)

# Example usage and visualization
if __name__ == "__main__":
    dataset = PriorDataset(n_samples=5, n_points=100)
    
    plt.figure()
    for i in range(5):
        u = dataset[i]
        plt.plot(dataset.x.cpu().numpy(), u.cpu().numpy()[0], label=f'Sample {i+1}')
    
    plt.title('Example Prior Functions')
    plt.legend()
    plt.show()