import os

import matplotlib.pyplot as plt
import torch
import yaml
from neural_operator import FNO
from noise import SpectralNoiseSampler
from prior import PriorDataset
from score_model import ScoreModel
from sde import OU
from tqdm import tqdm

with open("configs/base_model.yaml", "r") as f:
    config = yaml.safe_load(f)

power = config["model"]["power"]

load_dir = (
    f"unconditional_model_fno/model_type={config['model']['model_type']}/alpha={power}/"
)


device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = config["data"]["num_points"]

dataset = PriorDataset(n_samples=config["data"]["n_samples"], n_points=n_points)

model = FNO(
    modes=config["model"]["modes"],
    width=config["model"]["width"],
    n_layers=config["model"]["n_layers"],
    timestep_embedding_dim=config["model"]["timestep_embedding_dim"],
    max_period=config["model"]["max_period"],
)
model.load_state_dict(
    torch.load(
        os.path.join(load_dir, "ema_model.pt"), map_location=device, weights_only=False
    )["shadow"]
)
model.to(device)

print(
    "NUMBER OF PARAMETERS:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

num_epochs = config["training"]["num_epochs"]

optimizer = torch.optim.Adam(
    model.parameters(), lr=float(config["training"]["learning_rate"])
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=1
)

sde = OU(beta_min=config["sde"]["beta_min"], beta_max=config["sde"]["beta_max"])

noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)

pos = (
    torch.linspace(0, 1, n_points).to(device).unsqueeze(0).unsqueeze(0)
)  # (1,1,n_points)

score_model = ScoreModel(
    model=model,
    sde=sde,
    noise_sampler=noise_sampler,
    model_type=config["model"]["model_type"],
)


batch_size_smpl = 8
with torch.no_grad():
    num_timesteps = 1000
    ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
    delta_t = ts[1] - ts[0]

    xt = noise_sampler.sample(batch_size_smpl)  # N(0,C)

    for ti in tqdm(reversed(ts), total=len(ts)):
        t = torch.ones(batch_size_smpl).to(xt.device) * ti

        with torch.no_grad():
            pos_inp = pos.repeat(batch_size_smpl, 1, 1)  # (batch_size,1,n_points)
            score, x0hat = score_model(xt, t, pos_inp)

        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(batch_size_smpl)  # N(0,C)

        xt = (
            xt
            + beta_t / 2.0 * delta_t * xt
            + beta_t * delta_t * score
            + beta_t.sqrt() * delta_t.sqrt() * noise
        )

    fig, axes = plt.subplots(2, batch_size_smpl // 2, figsize=(12, 6))
    for idx, ax in enumerate(axes.ravel()):
        im = ax.plot(xt[idx, 0].cpu().numpy())
        ax.set_title("Samples")

    plt.savefig("unconditional_samples.png")
    plt.close()
