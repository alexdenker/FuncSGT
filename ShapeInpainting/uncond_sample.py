import os

import matplotlib.pyplot as plt
import torch
import yaml
from sde import OU
from simple_network import ScoreNet
from tqdm import tqdm
from utils import get_fourier_noise_scales, inverse_fourier

with open("config/base_model.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda"
num_fourier_modes = config["training"][
    "num_fourier_modes"
]  # Number of modes (will give 2*num_modes coefficients: pos + neg)

model = ScoreNet(
    input_dim=2 * num_fourier_modes * 2,
    output_dim=2 * num_fourier_modes * 2,
    hidden_dim=config["model"]["hidden_dim"],
    time_embed_dim=config["model"]["time_embed_dim"],
    depth=config["model"]["depth"],
    max_period=config["model"]["max_period"],
)
model.load_state_dict(
    torch.load("training_results/model_finite_dim_fourier.pt", weights_only=True)
)
model.to(device)
model.eval()


# Initialize the Fourier VP-SDE with frequency-dependent noise
sde = OU(beta_min=config["sde"]["beta_min"], beta_max=config["sde"]["beta_max"])

scales = get_fourier_noise_scales(
    num_bases=num_fourier_modes,
    scale_type="inv_k_sq",
    device=device,
    dtype=torch.float32,
)
scales = scales.view(1, 1, num_fourier_modes, 1)  # for broadcasting


def model_fun(xt, t):
    """Model predicts the score function"""
    pred = model(xt, t)
    std_t = sde.std_t_scaling(t, xt)
    return pred / std_t


batch_size_smpl = 5

landmarks_list = [32, 64, 128]
landmark_labels = ["32 Landmarks", "64 Landmarks", "128 Landmarks"]

# Create publication-ready figure for ICML (half-page width: 3.5 inches)
fig, axes = plt.subplots(
    len(landmarks_list), batch_size_smpl, figsize=(3.5, 3.2), dpi=300
)
fig.subplots_adjust(
    left=0.05, right=0.98, top=0.92, bottom=0.05, wspace=0.05, hspace=0.05
)

num_timesteps = 2000

for i, num_landmarks in enumerate(landmarks_list):
    print(f"Generating samples with {num_landmarks} landmarks...")

    # Sample from prior
    ts = torch.linspace(1e-3, 1, num_timesteps).to(device)

    delta_t = ts[1] - ts[0]
    xt_fourier = torch.sqrt(scales) * torch.randn(
        batch_size_smpl, 2, num_fourier_modes, 2, device=device
    )

    for ti in tqdm(reversed(ts), total=len(ts), leave=False):
        t = torch.ones(batch_size_smpl).to(xt_fourier.device) * ti

        with torch.no_grad():
            # Model prediction
            xt_flat = xt_fourier.reshape(batch_size_smpl, -1)
            score_flat = model_fun(xt_flat, t)
            score_fourier = score_flat.reshape(batch_size_smpl, 2, num_fourier_modes, 2)

        # Reshape for broadcasting
        beta_t = sde.beta_t(t).view(-1, 1, 1, 1)

        # Sample noise with proper scaling
        noise = torch.sqrt(scales) * torch.randn_like(xt_fourier)

        xt_fourier = (
            xt_fourier
            + beta_t / 2.0 * delta_t * xt_fourier
            + beta_t * delta_t * score_fourier
            + beta_t.sqrt() * delta_t.sqrt() * noise
        )

    # Convert back to spatial domain
    xt = inverse_fourier(xt_fourier, num_pts=num_landmarks)

    for idx in range(batch_size_smpl):
        ax = axes[i, idx]

        # Plot shape as closed contour with small markers
        ax.plot(
            xt[idx, :, 0].cpu().numpy(),
            xt[idx, :, 1].cpu().numpy(),
            "o-",
            linewidth=0.8,
            markersize=2,
            color="#1f77b4",
        )

        # Close the contour
        ax.plot(
            [xt[idx, -1, 0].item(), xt[idx, 0, 0].item()],
            [xt[idx, -1, 1].item(), xt[idx, 0, 1].item()],
            "o-",
            linewidth=0.8,
            markersize=2,
            color="#1f77b4",
        )

        ax.set_aspect("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])

        # Remove axis ticks and labels for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Add row label on the left
    axes[i, 0].text(
        -0.35,
        0.5,
        landmark_labels[i],
        transform=axes[i, 0].transAxes,
        fontsize=5,
        verticalalignment="center",
        rotation="vertical",
        weight="bold",
    )

# fig.suptitle('Generated MNIST Digit 3 Shapes with Score-Based Diffusion', fontsize=9, y=0.98)

# Save as publication-quality PDF and PNG
plt.savefig("uncond_shape_samples.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig("uncond_shape_samples.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("Saved: uncond_shape_samples.pdf and uncond_shape_samples.png")
plt.show()
