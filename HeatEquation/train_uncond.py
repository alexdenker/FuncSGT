
import os 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
import yaml 

from prior import PriorDataset
from sde import OU
from noise import SpectralNoiseSampler
from neural_operator import FNO
from score_model import ScoreModel
from ema import EMA 


with open("configs/base_model.yaml", 'r') as f:
    config = yaml.safe_load(f)

power = config['model']['power']

save_path = f"unconditional_model_fno/model_type={config['model']['model_type']}/alpha={power}/"
os.makedirs(save_path, exist_ok=True)

save_path_imgs = os.path.join(save_path, "imgs/")
os.makedirs(save_path_imgs, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = config['data']['num_points']

dataset = PriorDataset(n_samples=config['data']['n_samples'], n_points=n_points)

model = FNO(modes=config["model"]["modes"], 
            width=config["model"]["width"], 
            n_layers=config["model"]["n_layers"], 
            timestep_embedding_dim=config["model"]["timestep_embedding_dim"], 
            max_period=config["model"]["max_period"])
model.to(device)

num_epochs = config['training']['num_epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=1)

sde = OU(beta_min=config['sde']['beta_min'], beta_max=config['sde']['beta_max'])

noise_sampler = SpectralNoiseSampler(n=n_points, power=power, device=device)

pos = torch.linspace(0, 1, n_points).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,n_points)

score_model = ScoreModel(model=model, 
                         sde=sde, 
                         noise_sampler=noise_sampler, 
                         model_type=config['model']['model_type'])

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
        pos_inp = pos.repeat(x0.shape[0], 1, 1)  # (batch_size,1,n_points)
        pred, x0_hat = score_model(xt, random_t, pos_inp)

        residual = pred + z / std_t        
        residual = residual * std_t
        residual = noise_sampler.apply_Csqrt_inv(residual)

        loss = torch.sum(residual**2) / pred.shape[0]

        loss.backward() 
        mean_loss.append(loss.item())

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() 
        
        # Update EMA
        ema.update()
    print(f"Mean Loss: {np.mean(mean_loss)}")
    lr_scheduler.step()
    
    # Save both model and EMA
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(ema.state_dict(), os.path.join(save_path, "ema_model.pt"))

    if (epoch + 1) % 200 == 0:
        # sample with EMA model
        model.eval()
        ema.apply_shadow()  # Use EMA parameters for sampling

        batch_size_smpl = 8
        with torch.no_grad():
            num_timesteps = 100
            ts = torch.linspace(1e-3, 1.0, num_timesteps).to(device)
            delta_t = ts[1] - ts[0]

            xt = noise_sampler.sample(batch_size_smpl) # N(0,C)

            for ti in tqdm(reversed(ts), total=len(ts)):
                t = torch.ones(batch_size_smpl).to(xt.device)* ti

                with torch.no_grad():
                    pos_inp = pos.repeat(batch_size_smpl, 1, 1)  # (batch_size,1,n_points)
                    score, x0hat = score_model(xt, t, pos_inp) 

                beta_t = sde.beta_t(t).view(-1, 1, 1)
                noise = noise_sampler.sample(batch_size_smpl) # N(0,C)

                xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise

            fig, axes = plt.subplots(2, batch_size_smpl // 2, figsize=(12,6))
            for idx, ax in enumerate(axes.ravel()):
                im = ax.plot(xt[idx,0].cpu().numpy())
                ax.set_title("Samples")

            plt.savefig(os.path.join(save_path_imgs, f"samples_epoch_{epoch+1}.png"))
            plt.close()
        
        # Restore original parameters
        ema.restore()