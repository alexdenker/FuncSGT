import torch
import numpy as np 
from pathlib import Path
import json

import matplotlib.pyplot as plt
from tqdm import tqdm 
import yaml 
import argparse

from neural_operator import FNO
from sde import OU
from noise import SpectralNoiseSampler
from heat_equation import HeatEquation1D
from score_model import ScoreModel

argparse = argparse.ArgumentParser(description="Conditional Sampling with Score-Based Model")

#argparse.add_argument('--mask_ratio', type=float, default=0.4, help='Fraction of points to mask')
argparse.add_argument('--lambd', type=float, default=1.0, help='Guidance strength')
argparse.add_argument('--num_samples', type=int, default=200, help='Number of samples K to draw per measurement')
argparse.add_argument('--num_dataset_elements', type=int, default=64, help='Number of elements N to process from dataset')    
argparse.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
argparse.add_argument('--num_timesteps', type=int, default=1000, help='Number of timesteps for sampling')

def main(args):

    with open("configs/forward_op.yaml", 'r') as f:
        forward_op_config = yaml.safe_load(f)

    with open("configs/base_model.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Configuration
    device = args.device

    #mask_ratio = args.mask_ratio  # fraction of points to mask
    lambd = args.lambd  # guidance strength
    # Sampling parameters
    num_samples = args.num_samples  # Number of samples K to draw per measurement
    num_dataset_elements = args.num_dataset_elements  # Number of elements N to process from dataset
    n_points = 128
    # Create output directory for logging
    output_dir = Path(f"results/DPS/lambda_{lambd}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Load model
    model = FNO(modes=config['model']['modes'], 
                width=config['model']['width'], 
                n_layers=config['model']['n_layers'], 
                timestep_embedding_dim=config['model']['timestep_embedding_dim'], 
                max_period=config['model']['max_period'])

    model.to(device)
    model.load_state_dict(torch.load("unconditional_model_fno/model_type=C/alpha=1.0/ema_model.pt", map_location=device, weights_only=False)["shadow"])
    model.to(device)
    model.eval()

    # Initialize the Fourier VP-SDE with frequency-dependent noise
    sde = OU(beta_min=config['sde']['beta_min'], beta_max=config['sde']['beta_max'])
    power = config['model']['power']
    noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)
    model_type = config['model']['model_type']  #"C", "raw", "C_sqrt"
    num_timesteps = args.num_timesteps  # number of discretization steps for sampling

    score_model = ScoreModel(model=model,
                                sde=sde,
                                noise_sampler=noise_sampler,
                                model_type=model_type)


    def sample_conditional(x_gt, y, forward_op, num_samples_K,pos):
        """
        Generate K conditional samples for a given measurement y.
        
        Args:
            x_gt: Ground truth shape [1, n_points]
            y: Observed measurements [1, n_points]
            num_samples_K: Number of samples to generate
        
        Returns:
            samples: Generated samples [num_samples_K, n_points]
        """

        ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
        delta_t = ts[1] - ts[0]

        xt = noise_sampler.sample(num_samples_K) # N(0,C)

        y = y.repeat(num_samples_K, 1)  # shape: [num_samples_K, ny, 1]

        for ti in tqdm(reversed(ts), total=len(ts)):
            t = torch.ones(num_samples_K).to(xt.device)* ti

            xt.requires_grad_(True)
            pos_inp = pos.repeat(num_samples_K, 1, 1)  # (batch_size,1,n_points)
            score, x0hat = score_model(xt, t, pos_inp)

            y_pred = forward_op(x0hat.squeeze(1))  # shape: [num_samples_K, nx]
            loss = torch.sum((y_pred - y)**2)  # shape: [num_samples_K]
            grad = torch.autograd.grad(loss, xt)[0]
            
            #print("Grad norm:", grad.norm().item(), "Score norm:", score.norm().item())
            grad = noise_sampler.apply_C(grad)
            grad = grad / loss.sqrt().unsqueeze(-1).unsqueeze(-1)  # normalize gradient
            score = score #- lambd * grad  # Adjusted score with guidance

            beta_t = sde.beta_t(t).view(-1, 1, 1)
            noise = noise_sampler.sample(num_samples_K)# N(0,C)

            xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise
            xt = xt - lambd * grad
            xt = xt.detach()  # detach to avoid gradient accumulation

        return xt

    solver = HeatEquation1D(nx=n_points, 
                            nu=forward_op_config['nu'], 
                            dt=forward_op_config['dt'], 
                            t_max=forward_op_config['t_max'], 
                            device=device)


    # Load dataset
    torch.manual_seed(0)

    print(f"Processing {num_dataset_elements} dataset elements, generating {num_samples} samples each")
    print("=" * 80)

    # Results storage for logging
    results = []

    x_gts = torch.from_numpy(np.load("test_data/ground_truth.npy")).to(device)
    noisy_observations = torch.from_numpy(np.load("test_data/noisy_observations.npy")).to(device)

    for dataset_idx in range(num_dataset_elements):
        print(f"\n[{dataset_idx+1}/{num_dataset_elements}] Processing dataset element {dataset_idx}")
        
        # Load ground truth
        x_gt = x_gts[dataset_idx].unsqueeze(0) # shape: [1, nx]
        y_noise = noisy_observations[dataset_idx].unsqueeze(0)  # shape: [1, nx]

        # Generate K samples
        pos = solver.x_grid.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,nx)
        samples = sample_conditional(x_gt, y_noise, solver, num_samples,pos).squeeze(1)
        
        print("Samples:", samples.shape)

        # Compute metrics
        mse_all = torch.sum((samples - x_gt)**2).cpu().numpy()/samples.shape[1]
        #mse_obs = torch.mean((samples[:, mask, :] - x_gt[:, mask, :])**2, dim=[1, 2]).cpu().numpy()
        
        print(f"  - MSE : mean={mse_all.mean():.6f}, std={mse_all.std():.6f}")
        
        # Store results for this element
        element_results = {
            'dataset_idx': int(dataset_idx),
            'mse_all_mean': float(mse_all.mean()),
            'mse_all_std': float(mse_all.std()),
            'samples_shape': [int(s) for s in samples.shape],
        }
        results.append(element_results)
        
        # Save visualizations and data for this element
        element_dir = output_dir / f"element_{dataset_idx:04d}"
        element_dir.mkdir(exist_ok=True)
        
        # Save data as numpy arrays
        np.save(element_dir / "ground_truth.npy", x_gt.cpu().numpy())
        np.save(element_dir / "samples.npy", samples.cpu().numpy())
        np.save(element_dir / "noisy_data.npy", y_noise.cpu().numpy())
        np.save(element_dir / "mse_all.npy", mse_all)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(9,5))
        
        # Plot ground truth
        ax.plot(solver.x_grid.cpu().numpy(), x_gt[0,:].cpu().numpy(), c='k', label="ground truth")
        ax.plot(solver.x_grid.cpu().numpy(), y_noise[0,:].cpu().numpy(), c="r", label="observed")
        
        # Plot samples
        for sample_idx in range(num_samples):
            if sample_idx == 0:
                ax.plot(solver.x_grid.cpu().numpy(), samples[sample_idx,:].cpu().numpy(), c='g', alpha=0.01, label="samples")
            else:
                ax.plot(solver.x_grid.cpu().numpy(), samples[sample_idx,:].cpu().numpy(), c='g', alpha=0.01)
        
        # Plot mean and std with filled area
        avg_sample = samples.mean(dim=0)
        std_sample = samples.std(dim=0)
        x_grid_np = solver.x_grid.cpu().numpy()
        avg_np = avg_sample.cpu().numpy()
        std_np = std_sample.cpu().numpy()
        
        # Fill between mean ± std
        ax.fill_between(x_grid_np, avg_np - std_np, avg_np + std_np, 
                        color='b', alpha=0.3, label="mean ± std")
        # Plot mean
        ax.plot(x_grid_np, avg_np, c='b', alpha=0.9, linewidth=2, label="mean")
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(element_dir / "samples_visualization.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  - Saved results to {element_dir}")

    # Save summary statistics
    all_mse_all = np.array([r['mse_all_mean'] for r in results])

    summary = {
        'config': {
            'guidance_strength': lambd,
            'num_samples_per_measurement': num_samples,
            'num_dataset_elements_processed': num_dataset_elements,
            'mean_mse_all': float(all_mse_all.mean()),
            'std_mse_all': float(all_mse_all.std()),
        },
        'results': results,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print overall statistics
    print("\n" + "=" * 80)
    print("Overall Statistics:")
    all_mse_all = np.array([r['mse_all_mean'] for r in results])
    print(f"Mean MSE across all elements: {all_mse_all.mean():.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    args = argparse.parse_args()
    main(args)