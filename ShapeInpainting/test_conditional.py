"""
Conditional training of score-based model on Fourier coefficients of shapes.
The conditional information are the first K Fourier coefficients.

"""

import torch
import numpy as np 
import os 

import matplotlib.pyplot as plt
from tqdm import tqdm 

from dataset import MNISTShapesDataset
from simple_network import ScoreNet 
from utils import fourier_coefficients, inverse_fourier, get_fourier_noise_scales
from sde import OU
import json 
from pathlib import Path


num_cond_coeffs = 8  # number of Fourier coefficients to condition on


device = "cuda"
batch_size = 128 
batch_size_smpl = 16

num_landmarks = 64
num_fourier_modes = 16  # Number of modes (will give 2*num_modes coefficients: pos + neg)
num_dataset_elements = 10 
num_timesteps = 1000
num_samples = 5 
#from triangles_dataset import create_triangle_dataset

model = ScoreNet(input_dim=2*num_fourier_modes*2 + 2*num_cond_coeffs*2, 
                 output_dim=2*num_fourier_modes*2,
                 hidden_dim=1024, 
                 time_embed_dim=64, 
                 depth=12, 
                 max_period=2.0)
model.load_state_dict(torch.load(os.path.join(f"conditional_training_results/num_cond_coeffs_{num_cond_coeffs}/", "cond_model.pt")))
model.to(device)
model.eval()

# Initialize the Fourier VP-SDE with frequency-dependent noise
sde = OU(beta_min=0.001, beta_max=20.0)

scales = get_fourier_noise_scales(num_bases=num_fourier_modes, scale_type="inv_k_sq", device=device, dtype=torch.float32)
scales = scales.view(1, 1, num_fourier_modes, 1)  # for broadcasting

dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks, train=False)
num_dataset_elements = min(num_dataset_elements, len(dataset))


def model_fun(xt, t):
    """Model predicts the score function"""
    pred = model(xt, t)
    std_t = sde.std_t_scaling(t, xt)
    return pred / std_t

def sample_conditional(x_gt, y, num_samples_K):
    """
    Generate K conditional samples for a given measurement y.
    
    Args:
        x_gt: Ground truth shape [1, num_landmarks, 2]
        y: Observed landmarks [1, num_observed, 2]
        num_samples_K: Number of samples to generate
    
    Returns:
        samples: Generated samples [num_samples_K, num_landmarks, 2]
    """
    ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
    delta_t = ts[1] - ts[0]
    
    # Initialize K samples in batch
    x_init = torch.sqrt(scales) * torch.randn(num_samples_K, 2, num_fourier_modes, 2, device=device)
    xt = x_init.clone()

    y_batch = torch.repeat_interleave(y, num_samples_K, dim=0)

    for ti in tqdm(reversed(ts), total=len(ts), desc="Sampling", leave=False):
        t = torch.ones((num_samples_K,)).to(xt.device) * ti
        
        
        xt_flat = xt.reshape(num_samples_K, -1)
        model_input = torch.cat([xt_flat, y_batch], dim=1)
        score_flat = model_fun(model_input, t)
        score = score_flat.reshape(num_samples_K, 2, num_fourier_modes, 2)

        
        beta_t = sde.beta_t(t).view(-1, 1, 1, 1)
        noise = torch.sqrt(scales) * torch.randn_like(xt)

        xt = xt + beta_t/2.0 * delta_t*xt + beta_t * delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise
        xt = xt.detach()
    
    # Convert back to landmark space
    samples = inverse_fourier(xt, num_pts=num_landmarks)
    return samples

# Create output directory for logging
output_dir = Path(f"results/Cond/num_cond_coeffs_{num_cond_coeffs}")
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output directory: {output_dir}")


# Load dataset
dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks, train=False)
num_dataset_elements = min(num_dataset_elements, len(dataset))

print(f"Processing {num_dataset_elements} dataset elements, generating {num_samples} samples each")
print("=" * 80)

# Results storage for logging
results = []

for dataset_idx in range(num_dataset_elements):
    print(f"\n[{dataset_idx+1}/{num_dataset_elements}] Processing dataset element {dataset_idx}")
    
    # Load ground truth
    x_gt = dataset[dataset_idx].to(device).unsqueeze(0)
    
    # Create mask
    torch.manual_seed(dataset_idx)  # Use dataset_idx for reproducibility per element
    #mask = torch.zeros(num_landmarks)
    #num_mask = int(num_landmarks * mask_ratio)
    #mask[torch.randperm(num_landmarks)[:num_mask]] = 1
    #mask = mask.bool().to(device)
    
    # Create noisy observations
    x_fourier = fourier_coefficients(x_gt, num_bases=num_fourier_modes)  # (batch_size, 2, num_fourier_modes, 2)
    y = x_fourier[:, :, :num_cond_coeffs, :]  # (batch_size, 2, num_cond_coeffs, 2)

    y_flat = y.reshape(1, -1)  # Flatten for conditioning
    #def forward_op(x):
    #    x_fourier = fourier_coefficients(x, num_bases=num_fourier_modes)
    #    return x_fourier[:, :, :num_cond_coeffs, :]

    print("  - Observed Fourier coefficients shape: ", y.shape)
    #y = x_gt[:, mask, :]
    #torch.manual_seed(dataset_idx)
    #y = y + 0.01 * torch.randn_like(y)  # add small noise to observed points
    
    print(f"  - Ground truth shape: {x_gt.shape}")
    
    # Generate K samples
    samples = sample_conditional(x_gt, y_flat, num_samples)
    
    # Compute metrics
    mse_all = torch.sum((samples - x_gt)**2, dim=[1, 2]).cpu().numpy()
    print("mse_all shape:", mse_all.shape)
    y_pred = fourier_coefficients(samples, num_bases=num_fourier_modes)[:, :, :num_cond_coeffs, :]

    #mse_obs = torch.mean((samples[:, mask, :] - x_gt[:, mask, :])**2, dim=[1, 2]).cpu().numpy()
    print("y_pred shape:", y_pred.shape, " y shape:", y.shape)
    mse_obs = torch.sum((y_pred - y)**2, dim=[1, 2, 3]).cpu().numpy()
    print(f"  - MSE (all points): mean={mse_all.mean():.6f}, std={mse_all.std():.6f}")
    print(f"  - MSE (observed): mean={mse_obs.mean():.6f}, std={mse_obs.std():.6f}")
    
    # Store results for this element
    element_results = {
        'dataset_idx': int(dataset_idx),
        'mse_all_mean': float(mse_all.mean()),
        'mse_all_std': float(mse_all.std()),
        'mse_observed_mean': float(mse_obs.mean()),
        'mse_observed_std': float(mse_obs.std()),
        'num_observed_points': int(y.shape[1]),
        'samples_shape': [int(s) for s in samples.shape],
    }
    results.append(element_results)
    
    # Save visualizations and data for this element
    element_dir = output_dir / f"element_{dataset_idx:04d}"
    element_dir.mkdir(exist_ok=True)
    
    # Save data as numpy arrays
    np.save(element_dir / "ground_truth.npy", x_gt.cpu().numpy())
    np.save(element_dir / "samples.npy", samples.cpu().numpy())
    np.save(element_dir / "observed_landmarks.npy", y.cpu().numpy())
    np.save(element_dir / "mse_all.npy", mse_all)
    np.save(element_dir / "mse_observed.npy", mse_obs)
    
    # Create visualization
    fig, axes = plt.subplots(1, num_samples + 2, figsize=(3*(num_samples+2), 3))
    
    # Plot ground truth
    axes[0].plot(x_gt[0,:,0].cpu().numpy(), x_gt[0,:,1].cpu().numpy(), '-o', c='k', alpha=0.4, label="ground truth")
    #axes[0].scatter(y[0,:,0].cpu().numpy(), y[0,:,1].cpu().numpy(), c="r", s=10, zorder=5, label="observed")
    axes[0].set_title("Ground Truth + Observations")
    axes[0].legend()
    axes[0].axis("off")
    
    # Plot samples
    for sample_idx in range(num_samples):
        ax = axes[sample_idx + 1]
        ax.plot(x_gt[0,:,0].cpu().numpy(), x_gt[0,:,1].cpu().numpy(), '-o', c='k', alpha=0.3, label="ground truth")
        ax.plot(samples[sample_idx,:,0].cpu().numpy(), samples[sample_idx,:,1].cpu().numpy(), '-o', c='g', alpha=0.8, label="sample")
        #ax.scatter(y[0,:,0].cpu().numpy(), y[0,:,1].cpu().numpy(), c="r", s=10, zorder=5)
        ax.set_title(f"Sample {sample_idx+1}\nMSE all: {mse_all[sample_idx]:.4f}\nMSE obs: {mse_obs[sample_idx]:.4f}")
        ax.axis("off")
    
    # Plot average sample
    avg_sample = samples.mean(dim=0)
    axes[-1].plot(x_gt[0,:,0].cpu().numpy(), x_gt[0,:,1].cpu().numpy(), '-o', c='k', alpha=0.3, label="ground truth")
    axes[-1].plot(avg_sample[:,0].cpu().numpy(), avg_sample[:,1].cpu().numpy(), '-o', c='b', alpha=0.8, label="average")
    #axes[-1].scatter(y[0,:,0].cpu().numpy(), y[0,:,1].cpu().numpy(), c="r", s=10, zorder=5)
    axes[-1].set_title("Average of Samples")
    axes[-1].axis("off")
    
    plt.tight_layout()
    plt.savefig(element_dir / "samples_visualization.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  - Saved results to {element_dir}")

# Save summary statistics
all_mse_all = np.array([r['mse_all_mean'] for r in results])
all_mse_obs = np.array([r['mse_observed_mean'] for r in results])

summary = {
    'config': {
        'num_fourier_modes': num_fourier_modes,
        'num_timesteps': num_timesteps,
        'num_landmarks': num_landmarks,
        'num_samples_per_measurement': num_samples,
        'num_dataset_elements_processed': num_dataset_elements,
        'mean_mse_all': float(all_mse_all.mean()),
        'std_mse_all': float(all_mse_all.std()),
        'mean_mse_observed': float(all_mse_obs.mean()),
        'std_mse_observed': float(all_mse_obs.std()),
    },
    'results': results,
}

with open(output_dir / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Print overall statistics
print("\n" + "=" * 80)
print("Overall Statistics:")
all_mse_all = np.array([r['mse_all_mean'] for r in results])
all_mse_obs = np.array([r['mse_observed_mean'] for r in results])
print(f"Mean MSE (all points) across all elements: {all_mse_all.mean():.6f}")
print(f"Mean MSE (observed) across all elements: {all_mse_obs.mean():.6f}")
print(f"Results saved to: {output_dir}")
print("=" * 80)