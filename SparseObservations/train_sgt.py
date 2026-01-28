import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ema import EMA
from neural_operator import FNO, CondFNO
from noise import SpectralNoiseSampler
from point_evaluation import Conditioning, PointEvaluationOperator
from prior import PriorDataset
from score_model import HtransformModel
from sde import OU
from tqdm import tqdm

with open("configs/forward_op.yaml", "r") as f:
    forward_op_config = yaml.safe_load(f)

with open("configs/base_model.yaml", "r") as f:
    base_model_config = yaml.safe_load(f)

with open("configs/h_transform.yaml", "r") as f:
    h_transform_config = yaml.safe_load(f)

power = base_model_config["model"]["power"]
model_type = base_model_config["model"]["model_type"]  # "raw", "C_sqrt", "C"
cond_model_type = "C_sqrt"  # "raw"

save_path = f"h_transform/model_type={cond_model_type}/alpha={power}/"
os.makedirs(save_path, exist_ok=True)

save_path_imgs = os.path.join(save_path, "imgs/")
os.makedirs(save_path_imgs, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = base_model_config["data"]["num_points"]
conditioning_operator = Conditioning(
    nx=n_points, sigma=forward_op_config["cond_sigma"], device=device
)


dataset = PriorDataset(
    n_samples=base_model_config["data"]["n_samples"], n_points=n_points, start_seed=0
)

model = FNO(
    modes=base_model_config["model"]["modes"],
    width=base_model_config["model"]["width"],
    n_layers=base_model_config["model"]["n_layers"],
    timestep_embedding_dim=base_model_config["model"]["timestep_embedding_dim"],
    max_period=base_model_config["model"]["max_period"],
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

for param in model.parameters():
    param.requires_grad = False

h_trans = CondFNO(
    modes=h_transform_config["model"]["modes"],
    width=h_transform_config["model"]["width"],
    n_layers=h_transform_config["model"]["n_layers"],
    timestep_embedding_dim=h_transform_config["model"]["timestep_embedding_dim"],
    max_period=h_transform_config["model"]["max_period"],
)
h_trans.to(device)

num_epochs = h_transform_config["training"]["num_epochs"]
optimizer = torch.optim.Adam(
    h_trans.parameters(), lr=float(h_transform_config["training"]["learning_rate"])
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=h_transform_config["training"]["batch_size"],
    shuffle=True,
    num_workers=1,
)

sde = OU(
    beta_min=base_model_config["sde"]["beta_min"],
    beta_max=base_model_config["sde"]["beta_max"],
)
noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)

pos = (
    torch.linspace(0, 1, n_points).to(device).unsqueeze(0).unsqueeze(0)
)  # (1,1,n_points)

conditional_model = HtransformModel(
    h_trans=h_trans,
    model=model,
    sde=sde,
    noise_sampler=noise_sampler,
    model_type=model_type,
    cond_model_type=cond_model_type,
    cond_operator=conditioning_operator,
)

# Initialize EMA
ema_decay = base_model_config["training"].get("ema_decay", 0.999)
ema = EMA(h_trans, decay=ema_decay, device=device)
print(f"Using EMA with decay: {ema_decay}")

solver = PointEvaluationOperator(
    nx=n_points,
    n_points=forward_op_config["n_points"],
    sigma=forward_op_config["sigma"],
    seed=forward_op_config["seed"],
    device=device,
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    h_trans.train()

    mean_loss = []
    for idx, x in tqdm(enumerate(data_loader), total=len(data_loader)):

        optimizer.zero_grad()

        x0 = x.to(device)

        with torch.no_grad():
            y = solver(x0.squeeze(1))
            noise_std = forward_op_config["noise_std"]
            y = y + torch.randn_like(y) * noise_std

        random_t = torch.rand((x0.shape[0],), device=x0.device) * (1 - 0.001) + 0.001

        z = noise_sampler.sample(
            x0.shape[0]
        )  # N(0,C) # = C^{1/2} epsilon, espilon ~ N(0,I)

        mean_t = sde.mean_t(random_t, x0)
        std_t = sde.std_t_scaling(random_t, x0)
        xt = mean_t + std_t * z
        pos_inp = pos.repeat(x0.shape[0], 1, 1)  # (batch_size,1,n_points)
        pred, _ = conditional_model(
            x=xt,
            y=(solver.eval_points.unsqueeze(0).repeat(x0.shape[0], 1), y),
            grid=pos_inp,
            t=random_t,
            forward_op=solver,
        )
        residual = pred + z / std_t

        residual = residual * std_t
        residual = noise_sampler.apply_Csqrt_inv(residual)

        loss = torch.sum(residual**2) / pred.shape[0]

        loss.backward()
        mean_loss.append(loss.item())

        # clip gradients
        torch.nn.utils.clip_grad_norm_(h_trans.parameters(), max_norm=1.0)

        optimizer.step()
        ema.update()
    print(f"Mean Loss: {np.mean(mean_loss)}")
    lr_scheduler.step()
    torch.save(h_trans.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(ema.state_dict(), os.path.join(save_path, "ema_model.pt"))

    if (epoch + 1) % 200 == 0 or epoch == 0:
        # sample
        h_trans.eval()
        ema.apply_shadow()

        batch_size_smpl = 8
        with torch.no_grad():
            num_timesteps = 1000
            ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
            delta_t = ts[1] - ts[0]

            x0 = next(iter(data_loader)).to(device)
            x0 = x0[:batch_size_smpl]
            with torch.no_grad():
                y = solver(x0.squeeze(1))
                noise_std = forward_op_config["noise_std"]
                y = y + torch.randn_like(y) * noise_std

            xt = noise_sampler.sample(batch_size_smpl)  # N(0,C)

            for ti in tqdm(reversed(ts), total=len(ts)):
                t = torch.ones(batch_size_smpl).to(xt.device) * ti

                with torch.no_grad():
                    pos_inp = pos.repeat(
                        batch_size_smpl, 1, 1
                    )  # (batch_size,1,n_points)
                    score, _ = conditional_model(
                        x=xt,
                        y=(solver.eval_points.unsqueeze(0).repeat(x0.shape[0], 1), y),
                        grid=pos_inp,
                        t=t,
                        forward_op=solver,
                    )
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
                ax.plot(xt[idx, 0].cpu().numpy(), label="Sampled", linewidth=2)
                ax.plot(
                    x0[idx, 0].cpu().numpy(), "--", label="Ground Truth", linewidth=2
                )

            axes[0, 0].legend()

            plt.savefig(os.path.join(save_path_imgs, f"samples_epoch_{epoch+1}.png"))
            plt.close()
        ema.restore()
