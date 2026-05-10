
import os 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
import yaml 

from prior import PriorDataset
from sde import OU
from noise import WhiteNoiseSampler
from unet_1d import UNet1D
from score_model import ScoreModelFinDim
from ema import EMA
import torch.nn.functional as F

with open("configs/base_model.yaml", 'r') as f:
    config = yaml.safe_load(f)

power = config['model']['power']

n_points = 32

save_path = f"unconditional_model_finitedim/model_type={config['model']['model_type']}/alpha={power}/"
os.makedirs(save_path, exist_ok=True)

save_path_imgs = os.path.join(save_path, "imgs/")
os.makedirs(save_path_imgs, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PriorDataset(n_samples=config['data']['n_samples'], n_points=n_points, start_seed=0)

model = UNet1D(
    in_channels=1,
    base_channels=config["model"]["width"],
    channel_mults=(1, 2, 4),
    num_res_blocks=1,
    emb_dim=config["model"]["timestep_embedding_dim"],
    max_period=config["model"]["max_period"],
)
model.to(device)

num_epochs = config['training']['num_epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=1)

sde = OU(beta_min=config['sde']['beta_min'], beta_max=config['sde']['beta_max'])

noise_sampler = WhiteNoiseSampler(n=n_points, device=device)

n_points_ood = 64
noise_sampler_ood = WhiteNoiseSampler(n=n_points_ood, device=device)


score_model = ScoreModelFinDim(model=model, 
                               sde=sde, 
                               noise_sampler=noise_sampler)

# Initialize EMA
ema_decay = config['training'].get('ema_decay', 0.999)
ema = EMA(model, decay=ema_decay, device=device)
print(f"Using EMA with decay: {ema_decay}")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()

    mean_loss = []
    for idx, x in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()

        x0 = x.to(device)

        random_t = torch.rand((x0.shape[0],), device=x0.device) * (1-0.001) + 0.001

        z = noise_sampler.sample(x0.shape[0]) # N(0,C) # = C^{1/2} epsilon, espilon ~ N(0,I)

        mean_t = sde.mean_t(random_t, x0) 
        std_t = sde.std_t_scaling(random_t, x0)
        xt = mean_t + std_t * z 
        pred, x0_hat = score_model(xt, random_t)

        residual = pred + z / std_t        
        residual = residual * std_t

        loss = torch.sum(residual**2) / pred.shape[0]

        loss.backward() 
        mean_loss.append(loss.item())

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() 
        ema.update()
    print(f"Mean Loss: {np.mean(mean_loss)}")
    lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(ema.state_dict(), os.path.join(save_path, "ema_model.pt"))

    if (epoch + 1) % 100 == 0:
        # sample 
        model.eval()
        ema.apply_shadow()

        batch_size_smpl = 8
        with torch.no_grad():
            num_timesteps = 1000
            ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
            delta_t = ts[1] - ts[0]

            xt = noise_sampler.sample(batch_size_smpl) # N(0,C)

            for ti in tqdm(reversed(ts), total=len(ts)):
                t = torch.ones(batch_size_smpl).to(xt.device)* ti

                with torch.no_grad():
                    score, x0hat = score_model(xt, t)

                beta_t = sde.beta_t(t).view(-1, 1, 1)
                noise = noise_sampler.sample(batch_size_smpl) # N(0,C)

                xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise

            fig, axes = plt.subplots(2, batch_size_smpl // 2, figsize=(12,6))
            for idx, ax in enumerate(axes.ravel()):
                im = ax.plot(xt[idx,0].cpu().numpy())
                ax.set_title("Samples (train discretisation)")

            plt.savefig(os.path.join(save_path_imgs, f"samples_{n_points}_epoch_{epoch+1}.png"))
            plt.close()

        ### also sample using a different discretization to check generalization
        # discretisation in x
        with torch.no_grad():
            ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
            delta_t = ts[1] - ts[0]

            xt = noise_sampler_ood.sample(batch_size_smpl) # N(0,C)

            for ti in tqdm(reversed(ts), total=len(ts)):
                t = torch.ones(batch_size_smpl).to(xt.device)* ti

                with torch.no_grad():
                    ## interpolate xt to the original discretisation before feeding into the model
                    #xt_interp = F.interpolate(xt, size=n_points, mode='linear', align_corners=True)
                    # print(xt.shape, xt_interp.shape)
                    # plt.figure()
                    # plt.plot(np.linspace(0, 1, xt.shape[2]), xt[0,0].cpu().numpy(), label='xt (ood discretisation)')
                    # plt.plot(np.linspace(0, 1, xt_interp.shape[2]), xt_interp[0,0].cpu().numpy(), label='xt_interp (interpolated to train discretisation)')
                    # plt.legend()                   
                    # plt.title('Interpolation Check')
                    # plt.savefig(os.path.join(save_path_imgs, f"interpolation_check_epoch_{epoch+1}.png"))
                    # plt.close()
                    #score, _ = score_model(xt_interp, t) 
                    score, _ = score_model(xt, t)
                    #score = F.interpolate(score, size=n_points_ood, mode='linear', align_corners=True) # interpolate score back to ood discretisation

                beta_t = sde.beta_t(t).view(-1, 1, 1)
                noise = noise_sampler_ood.sample(batch_size_smpl) # N(0,C)

                xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise

            fig, axes = plt.subplots(2, batch_size_smpl // 2, figsize=(12,6))
            for idx, ax in enumerate(axes.ravel()):
                im = ax.plot(xt[idx,0].cpu().numpy())
                ax.set_title("Samples (ood discretisation)")

            plt.savefig(os.path.join(save_path_imgs, f"samples_{n_points_ood}_epoch_{epoch+1}.png"))
            plt.close()

        ema.restore()