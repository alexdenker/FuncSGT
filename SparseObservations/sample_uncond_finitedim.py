import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from noise import WhiteNoiseSampler
from score_model import ScoreModelFinDim
from sde import OU
from unet_1d import UNet1D


def parse_points(points_str):
    return [int(p.strip()) for p in points_str.split(",") if p.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 7,
            "xtick.labelsize": 4.5,
            "ytick.labelsize": 4.5,
            "legend.fontsize": 5.5,
            "lines.linewidth": 0.8,
            "axes.linewidth": 0.4,
            "grid.linewidth": 0.3,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model(config, device, model_dir=None):
    model = UNet1D(
        in_channels=1,
        base_channels=config["model"]["width"],
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        emb_dim=config["model"]["timestep_embedding_dim"],
        max_period=config["model"]["max_period"],
    )
    model.to(device)

    power = config["model"]["power"]
    load_dir = model_dir or f"unconditional_model_finitedim/model_type={config['model']['model_type']}/alpha={power}/"
    load_path = os.path.join(load_dir, "ema_model.pt")

    ema_state = torch.load(load_path, map_location=device, weights_only=False)
    # EMA checkpoint stores {"shadow": {name: tensor, ...}, "decay": float}
    shadow = ema_state["shadow"]
    model.load_state_dict(shadow)
    model.eval()
    return model, power, load_path


def sample_unconditional(model, sde, noise_sampler, n_points, num_samples, num_timesteps, device):
    """
    Sample from the finite-dim unconditional model.

    The UNet1D is fully convolutional, so it can be evaluated at any resolution
    without interpolation.
    """
    score_model = ScoreModelFinDim(
        model=model,
        sde=sde,
        noise_sampler=noise_sampler,
    )

    ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
    delta_t = ts[1] - ts[0]

    xt = noise_sampler.sample(num_samples)  # N(0, I), shape [num_samples, 1, n_points]

    for ti in tqdm(reversed(ts), total=len(ts)):
        t = torch.ones(num_samples, device=xt.device) * ti

        with torch.no_grad():
            score, _ = score_model(xt, t)

        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(num_samples)

        xt = xt + beta_t / 2.0 * delta_t * xt + beta_t * delta_t * score + beta_t.sqrt() * delta_t.sqrt() * noise

    return xt


def plot_and_save(samples, out_dir, n_points):
    out_dir = os.path.join(out_dir, f"n_points_{n_points}")
    os.makedirs(out_dir, exist_ok=True)

    samples_np = samples.squeeze(1).detach().cpu().numpy()
    np.save(os.path.join(out_dir, "samples.npy"), samples_np)

    set_plot_style()
    fig_width = 3.2
    fig_height = 2.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    x = np.linspace(0.0, 1.0, n_points)
    num_show = min(4, samples_np.shape[0])
    for idx in range(num_show):
        ax.plot(x, samples_np[idx], alpha=0.7)
    y_min = np.min(samples_np[:num_show])
    y_max = np.max(samples_np[:num_show])
    y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
    y_rug = y_min - 0.6 * y_pad
    ax.scatter(x, np.full_like(x, y_rug), s=5, color="black", alpha=0.5, marker="|")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([y_min - 1.2 * y_pad, y_max + y_pad])
    ax.set_title(f"n={n_points}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "samples.png"))
    plt.savefig(os.path.join(out_dir, "samples.pdf"))
    plt.close()


def main(args):
    with open("configs/base_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = resolve_device(args.device)
    model, power, load_path = load_model(config, device, args.model_dir)

    print("Loaded model:", load_path)
    print("Device:", device)
    print("Power:", power)

    sde = OU(beta_min=config["sde"]["beta_min"], beta_max=config["sde"]["beta_max"])

    points_list = parse_points(args.points)
    points_list = sorted(set(points_list))

    for n_points in points_list:
        print(f"\nSampling for n_points={n_points}")
        set_seed(args.seed)

        noise_sampler = WhiteNoiseSampler(n=n_points, device=device)
        samples = sample_unconditional(
            model=model,
            sde=sde,
            noise_sampler=noise_sampler,
            n_points=n_points,
            num_samples=args.num_samples,
            num_timesteps=args.num_timesteps,
            device=device,
        )

        plot_and_save(samples, args.output_dir, n_points)
        print(f"  Saved results to {args.output_dir}/n_points_{n_points}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from finite-dim unconditional diffusion model (UNet1D) at multiple resolutions"
    )
    parser.add_argument("--points", type=str, default="32,64,128,256", help="Comma-separated list of discretizations")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per discretization")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of SDE timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (reused per discretization)")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--model_dir", type=str, default=None, help="Override model directory (contains ema_model.pt)")
    parser.add_argument("--output_dir", type=str, default="results/uncond_samples_finitedim", help="Output directory")
    parser.parse_args()  # validate
    main(parser.parse_args())
