import torch
import numpy as np 
from pathlib import Path
import json

import matplotlib.pyplot as plt
from tqdm import tqdm 
import yaml 
import argparse

from dataset import MNISTShapesDataset
from simple_network import ScoreNet 
from utils import inverse_fourier, get_fourier_noise_scales, fourier_coefficients
from sde import OU

argparse = argparse.ArgumentParser(description="Conditional Sampling with Score-Based Model for Shape Data")

#argparse.add_argument('--mask_ratio', type=float, default=0.4, help='Fraction of points to mask')
argparse.add_argument('--num_coeffs', type=int, default=5, help='Number of Fourier coefficients observed')
argparse.add_argument('--lambd', type=float, default=50.0, help='Guidance strength')
argparse.add_argument('--num_samples', type=int, default=5, help='Number of samples K to draw per measurement')
argparse.add_argument('--num_dataset_elements', type=int, default=10, help='Number of elements N to process from dataset')    
argparse.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
argparse.add_argument('--num_timesteps', type=int, default=1000, help='Number of timesteps for sampling')
argparse.add_argument('--part', type=str, default='test', help='Dataset part to use (train or test)')

def main(args):
    # Configuration
    device = args.device

    # load base model 
    with open("config/base_model.yaml", 'r') as f:
        config = yaml.safe_load(f)

    num_fourier_modes = config['training']['num_fourier_modes']
    num_landmarks = config['training']['num_landmarks']
    #mask_ratio = args.mask_ratio  # fraction of points to mask
    num_cond_coeffs = args.num_coeffs  # number of Fourier coefficients observed
    lambd = args.lambd  # guidance strength
    part = args.part  # dataset part: train or test
    # Sampling parameters
    num_samples = args.num_samples  # Number of samples K to draw per measurement
    num_dataset_elements = args.num_dataset_elements  # Number of elements N to process from dataset

    # Create output directory for logging
    output_dir = Path(f"results/DPS/{part}/num_cond_coeffs_{num_cond_coeffs}/lambda_{lambd}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Load model
    model = ScoreNet(input_dim=2*num_fourier_modes*2, 
                    output_dim=2*num_fourier_modes*2,
                    hidden_dim=config['model']['hidden_dim'], 
                    time_embed_dim=config['model']['time_embed_dim'], 
                    depth=config['model']['depth'], 
                    max_period=config['model']['max_period'])
    model.load_state_dict(torch.load("training_results/model_finite_dim_fourier.pt", weights_only=True))
    model.to(device)
    model.eval()

    # Initialize the Fourier VP-SDE with frequency-dependent noise
    sde = OU(beta_min=config['sde']['beta_min'], beta_max=config['sde']['beta_max'])

    scales = get_fourier_noise_scales(num_bases=num_fourier_modes, scale_type="inv_k_sq", device=device, dtype=torch.float32)
    scales = scales.view(1, 1, num_fourier_modes, 1)  # for broadcasting

    num_timesteps = args.num_timesteps  # number of discretization steps for sampling

    def model_fun(xt, t):
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
            
            xt = xt.requires_grad_(True)
            
            score = model_fun(xt.reshape(xt.shape[0], -1), t)
            score = score.reshape(xt.shape[0], 2, num_fourier_modes, 2)
            
            beta_t = sde.beta_t(t).view(-1, 1, 1, 1)
            noise = torch.sqrt(scales) * torch.randn_like(xt)
            
            # Compute gradient for guidance
            mean_t_scale = sde.mean_t_scaling(t, xt)
            std_t = sde.std_t_scaling(t, xt)
            
            x0hat = (xt + std_t**2 * score) / mean_t_scale
            
            y_pred = x0hat[:, :, :num_cond_coeffs, :] #forward_op(x0hat)

            #x0hat_landmarks = inverse_fourier(x0hat, num_pts=num_landmarks)
            #loss = torch.sum(((x0hat_landmarks[:, mask, :] - y_batch))**2)
            #loss = torch.sum(((x0hat[:,:, :y_batch.shape[2], :] - y_batch))**2)
            loss = torch.sum(((y_pred - y_batch))**2)
            grad = torch.autograd.grad(loss, xt)[0]
            
            score = score - lambd * grad / (loss.sqrt() + 1e-8)
            
            xt = xt + beta_t/2.0 * delta_t*xt + beta_t * delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 
            xt = xt.detach()
        
        # Convert back to landmark space
        samples = inverse_fourier(xt, num_pts=num_landmarks)
        return samples


    # Load dataset
    part = args.part
    if part == 'train':
        dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks, train=True)
    else:
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

        #def forward_op(x):
        #    x_fourier = fourier_coefficients(x, num_bases=num_fourier_modes)
        #    return x_fourier[:, :, :num_cond_coeffs, :]

        print("  - Observed Fourier coefficients shape: ", y.shape)
        #y = x_gt[:, mask, :]
        #torch.manual_seed(dataset_idx)
        #y = y + 0.01 * torch.randn_like(y)  # add small noise to observed points
        
        print(f"  - Ground truth shape: {x_gt.shape}")
        
        # Generate K samples
        samples = sample_conditional(x_gt, y, num_samples)
        
        # Compute metrics
        mse_all = torch.sum((samples - x_gt)**2, dim=[1, 2]).cpu().numpy()
        y_pred = fourier_coefficients(samples, num_bases=num_fourier_modes)[:, :, :num_cond_coeffs, :]
        #mse_obs = torch.mean((samples[:, mask, :] - x_gt[:, mask, :])**2, dim=[1, 2]).cpu().numpy()
        print("y_pred shape:", y_pred.shape)
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
            'guidance_strength': lambd,
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

if __name__ == "__main__":
    args = argparse.parse_args()
    main(args)