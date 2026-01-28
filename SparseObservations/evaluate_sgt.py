import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from neural_operator import FNO, CondFNO
from noise import SpectralNoiseSampler
from point_evaluation import Conditioning, PointEvaluationOperator
from score_model import HtransformModel
from sde import OU
from tqdm import tqdm

with open("configs/forward_op.yaml", "r") as f:
    forward_op_config = yaml.safe_load(f)

with open("configs/base_model.yaml", "r") as f:
    base_config = yaml.safe_load(f)

with open("configs/h_transform.yaml", "r") as f:
    h_transform_config = yaml.safe_load(f)

power = base_config["model"]["power"]

model_type = base_config["model"]["model_type"]
cond_model_type = "raw"  # "raw" "C"
load_path = f"unconditional_model_fno/model_type={model_type}/alpha={power}/"
load_path_htrans = f"h_transform/model_type={cond_model_type}/alpha={power}/"

save_path = Path(f"results/h_transform/model_type={cond_model_type}/alpha={power}")
save_path.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = base_config["data"]["num_points"]
num_timesteps = 1000

conditioning_operator = Conditioning(
    nx=n_points, sigma=forward_op_config["cond_sigma"], device=device
)


model = FNO(
    modes=base_config["model"]["modes"],
    width=base_config["model"]["width"],
    n_layers=base_config["model"]["n_layers"],
    timestep_embedding_dim=base_config["model"]["timestep_embedding_dim"],
    max_period=base_config["model"]["max_period"],
)
model.load_state_dict(
    torch.load(
        os.path.join(
            f"unconditional_model_fno/model_type={model_type}/alpha={power}/",
            "ema_model.pt",
        ),
        weights_only=False,
    )["shadow"]
)
model.to(device)
model.eval()

print(
    "NUMBER OF PARAMETERS in base model:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

for param in model.parameters():
    param.requires_grad = False

print("LOAD h-transform model from:", load_path_htrans)
h_trans = CondFNO(
    modes=h_transform_config["model"]["modes"],
    width=h_transform_config["model"]["width"],
    n_layers=h_transform_config["model"]["n_layers"],
    timestep_embedding_dim=h_transform_config["model"]["timestep_embedding_dim"],
    max_period=h_transform_config["model"]["max_period"],
)
h_trans.load_state_dict(
    torch.load(os.path.join(load_path_htrans, "ema_model.pt"), weights_only=False)[
        "shadow"
    ]
)
h_trans.to(device)
h_trans.eval()

### plot the trained scaling network

t_test = torch.linspace(0, 1, 100).to(device)
with torch.no_grad():
    scaling = h_trans.scaling_network(t_test).cpu().numpy()

plt.figure(figsize=(4, 3))
plt.plot(t_test.cpu().numpy(), scaling)
plt.xlabel("t")
plt.ylabel("scaling factor")
plt.title("Trained Scaling Network in h-transform Model")
plt.grid()
plt.tight_layout()
plt.savefig(save_path / "scaling_network.png", dpi=100)
plt.close()

sde = OU(
    beta_min=base_config["sde"]["beta_min"], beta_max=base_config["sde"]["beta_max"]
)
noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)

pos = (
    torch.linspace(0, 1, n_points).to(device).unsqueeze(0).unsqueeze(0)
)  # (1,1,n_points)

cond_model = HtransformModel(
    h_trans=h_trans,
    model=model,
    sde=sde,
    noise_sampler=noise_sampler,
    model_type=model_type,
    cond_model_type=cond_model_type,
    cond_operator=conditioning_operator,
)


def sample_conditional(y, num_samples_K, pos, forward_op):
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

    y = y.repeat(num_samples_K, 1)  # shape: [num_samples_K, ny, 1]
    pos_inp = pos.repeat(num_samples_K, 1, 1)  # (batch_size,1,n_points)

    print("y.shape in sample_conditional:", y.shape)
    for ti in tqdm(reversed(ts), total=len(ts)):
        t = torch.ones(num_samples_K).to(xt.device) * ti

        with torch.no_grad():
            score, _ = cond_model(
                x=xt,
                y=(forward_op.eval_points.unsqueeze(0).repeat(xt.shape[0], 1), y),
                t=t,
                grid=pos_inp,
                forward_op=forward_op,
            )

        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(num_samples_K)  # N(0,C)

        xt = (
            xt
            + beta_t / 2.0 * delta_t * xt
            + beta_t * delta_t * score
            + beta_t.sqrt() * delta_t.sqrt() * noise
        )

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
    samples = sample_conditional(
        y=y_noise, num_samples_K=num_samples, pos=pos, forward_op=solver
    ).squeeze(1)

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
