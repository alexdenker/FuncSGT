import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from neural_operator import ConditionalFNO
from noise import SpectralNoiseSampler
from point_evaluation import Conditioning, PointEvaluationOperator
from score_model import CondScoreModel
from sde import OU
from tqdm import tqdm


cond_model_small = True 

with open("configs/forward_op.yaml", 'r') as f:
    forward_op_config = yaml.safe_load(f)

if cond_model_small:
    with open("configs/conditional_model_small.yaml", 'r') as f:
        config = yaml.safe_load(f)
else:
    with open("configs/conditional_model.yaml", 'r') as f:
        config = yaml.safe_load(f)

power = config["model"]["power"]

model_type = config["model"]["model_type"]

if cond_model_small:
    load_path = f"conditional_model_small/model_type={model_type}/alpha={power}/"
    save_path = Path(f"results/conditional_model_small/model_type={model_type}/alpha={power}")
else:
    load_path = f"conditional_model/model_type={model_type}/alpha={power}/"
    save_path = Path(f"results/conditional_model/model_type={model_type}/alpha={power}")
save_path.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = config["data"]["num_points"]
num_timesteps = 1000

conditioning_operator = Conditioning(
    nx=n_points, sigma=forward_op_config["cond_sigma"], device=device
)


model = ConditionalFNO(
    modes=config["model"]["modes"],
    width=config["model"]["width"],
    n_layers=config["model"]["n_layers"],
    timestep_embedding_dim=config["model"]["timestep_embedding_dim"],
    max_period=config["model"]["max_period"],
)
model.load_state_dict(
    torch.load(os.path.join(load_path, "model.pt"), weights_only=True)
)
model.to(device)
model.eval()

print(
    "NUMBER OF PARAMETERS in model:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

sde = OU(beta_min=config["sde"]["beta_min"], beta_max=config["sde"]["beta_max"])
noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)

pos = (
    torch.linspace(0, 1, n_points).to(device).unsqueeze(0).unsqueeze(0)
)  # (1,1,n_points)


conditional_model = CondScoreModel(
    model=model,
    sde=sde,
    noise_sampler=noise_sampler,
    model_type=model_type,
    cond_operator=conditioning_operator,
)


def sample_conditional(y, num_samples_K, pos, forward_op: PointEvaluationOperator):
    """
    Generate K conditional samples for a given measurement y.

    Args:
        y: Observed measurements [1, n_points]
        num_samples_K: Number of samples to generate
        pos: Position grid for the model
        forward_op: Forward operator used for conditioning
    Returns:
        samples: Generated samples [num_samples_K, n_points]
    """

    ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
    delta_t = ts[1] - ts[0]

    xt = noise_sampler.sample(num_samples_K)  # N(0,C)
    # print("xt initial:", xt.shape)
    y = y.repeat(num_samples_K, 1)  # shape: [num_samples_K, ny, 1]
    pos_inp = pos.repeat(num_samples_K, 1, 1)  # (batch_size,1,n_points)
    # print("Observation y:", y.shape)
    for ti in tqdm(reversed(ts), total=len(ts)):
        t = torch.ones(num_samples_K).to(xt.device) * ti

        with torch.no_grad():
            score, _ = conditional_model(
                xt,
                (forward_op.eval_points.unsqueeze(0).repeat(xt.shape[0], 1), y),
                t,
                pos_inp,
            )
        # print("score:", score.shape)
        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(num_samples_K)  # N(0,C)
        # print("noise:", noise.shape)
        xt = (
            xt
            + beta_t / 2.0 * delta_t * xt
            + beta_t * delta_t * score
            + beta_t.sqrt() * delta_t.sqrt() * noise
        )
        # print("New xt:", xt.shape)
    return xt


num_dataset_elements = 64
num_samples = 200
# Load dataset
torch.manual_seed(0)

print(
    f"Processing {num_dataset_elements} dataset elements, generating {num_samples} samples each"
)
print("=" * 80)

# Results storage for logging
results = []
x_gts = torch.from_numpy(np.load("test_data/ground_truth.npy")).to(device)
noisy_observations = torch.from_numpy(np.load("test_data/noisy_observations.npy")).to(
    device
)
seeds = np.load("test_data/seeds.npy")
for dataset_idx in range(num_dataset_elements):
    print(
        f"\n[{dataset_idx+1}/{num_dataset_elements}] Processing dataset element {dataset_idx}"
    )

    solver = PointEvaluationOperator(
        nx=n_points,
        n_points=forward_op_config["n_points"],
        sigma=forward_op_config["sigma"],
        seed=int(seeds[dataset_idx]),
        device=device,
    )

    eval_points_np = solver.eval_points.cpu().numpy()

    # Load ground truth
    x_gt = x_gts[dataset_idx].unsqueeze(0)  # shape: [1, nx]
    y_noise = noisy_observations[dataset_idx].unsqueeze(0)  # shape: [1, nx]

    # Generate K samples
    pos = solver.x_grid.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,nx)
    samples = sample_conditional(y_noise, num_samples, pos, forward_op=solver).squeeze(
        1
    )

    print("Samples:", samples.shape)

    # Compute metrics
    mse_all = torch.sum((samples - x_gt) ** 2).cpu().numpy() / samples.shape[1]
    # mse_obs = torch.mean((samples[:, mask, :] - x_gt[:, mask, :])**2, dim=[1, 2]).cpu().numpy()

    print(f"  - MSE : mean={mse_all.mean():.6f}, std={mse_all.std():.6f}")

    # Store results for this element
    element_results = {
        "dataset_idx": int(dataset_idx),
        "mse_all_mean": float(mse_all.mean()),
        "mse_all_std": float(mse_all.std()),
        "samples_shape": [int(s) for s in samples.shape],
    }
    results.append(element_results)

    # Save visualizations and data for this element
    element_dir = save_path / f"element_{dataset_idx:04d}"
    element_dir.mkdir(exist_ok=True)

    # Save data as numpy arrays
    np.save(element_dir / "ground_truth.npy", x_gt.cpu().numpy())
    np.save(element_dir / "samples.npy", samples.cpu().numpy())
    np.save(element_dir / "noisy_data.npy", y_noise.cpu().numpy())
    np.save(element_dir / "mse_all.npy", mse_all)
    np.save(element_dir / "eval_points.npy", eval_points_np)
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Plot ground truth
    ax.plot(
        solver.x_grid.cpu().numpy(),
        x_gt[0, :].cpu().numpy(),
        c="k",
        label="ground truth",
    )
    ax.scatter(
        solver.eval_points.cpu().numpy(),
        y_noise.cpu().numpy().flatten(),
        color="red",
        s=20,
        label="Measurement Points",
        zorder=5,
    )

    # Plot samples
    for sample_idx in range(num_samples):
        if sample_idx == 0:
            ax.plot(
                solver.x_grid.cpu().numpy(),
                samples[sample_idx, :].cpu().numpy(),
                c="g",
                alpha=0.01,
                label="samples",
            )
        else:
            ax.plot(
                solver.x_grid.cpu().numpy(),
                samples[sample_idx, :].cpu().numpy(),
                c="g",
                alpha=0.01,
            )

    # Plot mean and std with filled area
    avg_sample = samples.mean(dim=0)
    std_sample = samples.std(dim=0)
    x_grid_np = solver.x_grid.cpu().numpy()
    avg_np = avg_sample.cpu().numpy()
    std_np = std_sample.cpu().numpy()

    # Fill between mean ± std
    ax.fill_between(
        x_grid_np,
        avg_np - std_np,
        avg_np + std_np,
        color="b",
        alpha=0.3,
        label="mean ± std",
    )
    # Plot mean
    ax.plot(x_grid_np, avg_np, c="b", alpha=0.9, linewidth=2, label="mean")

    ax.legend()
    plt.tight_layout()
    plt.savefig(element_dir / "samples_visualization.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"  - Saved results to {element_dir}")

# Save summary statistics
all_mse_all = np.array([r["mse_all_mean"] for r in results])

summary = {
    "config": {
        "num_samples_per_measurement": num_samples,
        "num_dataset_elements_processed": num_dataset_elements,
        "mean_mse_all": float(all_mse_all.mean()),
        "std_mse_all": float(all_mse_all.std()),
    },
    "results": results,
}

with open(save_path / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print overall statistics
print("\n" + "=" * 80)
print("Overall Statistics:")
all_mse_all = np.array([r["mse_all_mean"] for r in results])
print(f"Mean MSE across all elements: {all_mse_all.mean():.6f}")
print("=" * 80)
