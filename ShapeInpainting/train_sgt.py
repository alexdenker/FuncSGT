import torch
import numpy as np 
import os 
import yaml 

import matplotlib.pyplot as plt
from tqdm import tqdm 

from dataset import MNISTShapesDataset
from simple_network import ScoreNet, CondScoreNet
from utils import fourier_coefficients, inverse_fourier, get_fourier_noise_scales
from sde import OU

num_cond_coeffs = 8  # number of Fourier coefficients to condition on

save_dir = f"h_transform/num_cond_coeffs_{num_cond_coeffs}/"
os.makedirs(save_dir, exist_ok=True)


save_image_path = os.path.join(save_dir, "imgs/")
os.makedirs(save_image_path, exist_ok=True)

device = "cuda"

with open("config/base_model.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open("config/h_transform.yaml", 'r') as f:
    config_htransform = yaml.safe_load(f)


batch_size = config['training']['batch_size']
batch_size_smpl = config['training']['batch_size_smpl']
num_epochs = config['training']['num_epochs']

num_landmarks = config['training']['num_landmarks']
num_fourier_modes = config['training']['num_fourier_modes']  

dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks)

model = ScoreNet(input_dim=2*num_fourier_modes*2, 
                 output_dim=2*num_fourier_modes*2,
                 hidden_dim=config['model']['hidden_dim'], 
                 time_embed_dim=config['model']['time_embed_dim'], 
                 depth=config['model']['depth'], 
                 max_period=config['model']['max_period'])
model.load_state_dict(torch.load("training_results/model_finite_dim_fourier.pt", weights_only=True))
model.to(device)
model.eval()

cond_model = CondScoreNet(input_dim=2*num_fourier_modes*2 + 2*num_cond_coeffs*2, 
                      output_dim=2*num_fourier_modes*2,
                      hidden_dim=config_htransform['model']['hidden_dim'], 
                      time_embed_dim=config_htransform['model']['time_embed_dim'], 
                      depth=config_htransform['model']['depth'], 
                      max_period=config_htransform['model']['max_period'])
cond_model.to(device)
cond_model.train()



optimizer = torch.optim.Adam(cond_model.parameters(), lr=float(config['training']['learning_rate']))

# Initialize the Fourier VP-SDE with frequency-dependent noise
sde = OU(beta_min=config['sde']['beta_min'], beta_max=config['sde']['beta_max'])

scales = get_fourier_noise_scales(num_bases=num_fourier_modes, scale_type="inv_k_sq", device=device, dtype=torch.float32)
scales = scales.view(1, 1, num_fourier_modes, 1)  # for broadcasting

data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=6)

def model_fun(xt, t):
    """Model predicts the score function"""
    pred = model(xt, t)
    std_t = sde.std_t_scaling(t, xt)
    return pred / std_t

def cond_model_fun(xt, t, cond):
    """Conditional model predicts the score function"""
    #print("xt shape:", xt.shape, " cond shape:", cond.shape)
    inp = torch.cat([xt, cond], dim=-1)

    with torch.enable_grad():
        xt.requires_grad_(True)
        xt_reshape = xt.reshape(xt.shape[0], 2, num_fourier_modes, 2)
        y_pred = xt_reshape[:, :, :num_cond_coeffs, :]  # Extract predicted conditioning coefficients
        y_pred = y_pred.reshape(xt.shape[0], -1)
        # Compute log likelihood gradient for conditioning
        loss_y = torch.mean((y_pred - cond)**2)  # MSE loss
        log_likelihood_grad = torch.autograd.grad(loss_y.sum(), xt, retain_graph=True)[0].detach()

    pred = cond_model(inp, log_likelihood_grad, t)
    std_t = sde.std_t_scaling(t, xt)
    return pred / std_t

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    mean_loss = [] 
    model.train() 
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()

        x = batch.to(device) # get my MNIST landmarks (batch_size, num_landmarks, 2)
        
        # Convert to Fourier space
        x_fourier = fourier_coefficients(x, num_bases=num_fourier_modes)  # (batch_size, 2, num_fourier_modes, 2)
        cond_fourier = x_fourier[:, :, :num_cond_coeffs, :]  # (batch_size, 2, num_cond_coeffs, 2)
        cond_flat = cond_fourier.reshape(x.shape[0], -1)

        # Sample random times
        random_t = torch.rand((x.shape[0],), device=x.device) * (1 - 1e-3) + 1e-3
        z = torch.sqrt(scales) * torch.randn_like(x_fourier)  # scaled noise
        # Add noise using the FourierVPSDE
        mean_t = sde.mean_t(random_t, x_fourier)
        std_t = sde.std_t_scaling(random_t, x_fourier)

        xt = mean_t + std_t * z  # noised data
        
        # Flatten for model input
        xt_flat = xt.reshape(x.shape[0], -1)  # flatten for model input

        # Model prediction
        with torch.no_grad():
            pred_flat = model_fun(xt_flat, random_t)
        pred_fourier = pred_flat.reshape(x.shape[0], 2, num_fourier_modes, 2)

        cond_pred = cond_model_fun(xt_flat, random_t, cond_flat)
        cond_pred_fourier = cond_pred.reshape(x.shape[0], 2, num_fourier_modes, 2)

        pred = pred_fourier + cond_pred_fourier
        res = pred + z / std_t

        # MSE loss in Fourier space
        loss = torch.mean((res) ** 2)
        loss.backward() 

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(cond_model.parameters(), max_norm=1.0)

        optimizer.step() 

        mean_loss.append(loss.item())

    print("Mean loss: ", np.mean(mean_loss))
    if epoch % 50 == 0 and epoch > 0:
        torch.save(cond_model.state_dict(), os.path.join(save_dir, "h_transform.pt"))

        # Sample from prior
        batch_size_smpl = 8
        x_gt = next(iter(data_loader)).to(device)
        x_gt = x_gt[:batch_size_smpl]
        cond_fourier = fourier_coefficients(x_gt, num_bases=num_fourier_modes)[:, :, :num_cond_coeffs, :]
        cond_flat = cond_fourier.reshape(batch_size_smpl, -1)        
        
        num_timesteps = 1000
        ts = torch.linspace(1e-3, 1, num_timesteps).to(device)

        delta_t = ts[1] - ts[0]
        xt_fourier = torch.sqrt(scales) * torch.randn(batch_size_smpl, 2, num_fourier_modes, 2, device=device)

        for ti in tqdm(reversed(ts), total=len(ts)):
            t = torch.ones(batch_size_smpl).to(xt_fourier.device) * ti

            with torch.no_grad():
                # Model prediction
                xt_flat = xt_fourier.reshape(batch_size_smpl, -1)
                score_flat = model_fun(xt_flat, t)
                score_fourier = score_flat.reshape(batch_size_smpl, 2, num_fourier_modes, 2)
                cond_score_flat = cond_model_fun(xt_flat, t, cond_flat)
                cond_score_fourier = cond_score_flat.reshape(batch_size_smpl, 2, num_fourier_modes, 2)
                score_fourier = score_fourier + cond_score_fourier
            # Get noise scaling and beta schedule
            beta_t = sde.beta_t(t).view(-1, 1, 1)
            
            # Reshape for broadcasting
            beta_t = beta_t.view(-1, 1, 1, 1)
            
            # Sample noise with proper scaling
            noise = torch.sqrt(scales) * torch.randn_like(xt_fourier)

            xt_fourier = xt_fourier + beta_t/2.0 * delta_t*xt_fourier + beta_t* delta_t * score_fourier + beta_t.sqrt()*delta_t.sqrt() * noise 

        # Convert back to spatial domain
        xt = inverse_fourier(xt_fourier, num_pts=num_landmarks)
        
        print("samples: ", xt.shape)
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))

        for idx in range(4):
            ax = axes[0, idx]
            ax.plot(x_gt[idx, :, 0].cpu().numpy(), x_gt[idx, :, 1].cpu().numpy(), '-o')
            ax.set_title(f"Ground Truth {idx + 1}")

            ax = axes[1, idx]
            ax.plot(xt[idx, :, 0].cpu().numpy(), xt[idx, :, 1].cpu().numpy(), '-o')
            ax.set_title(f"Sample {idx + 1}")


        plt.savefig(os.path.join(save_image_path, f"samples_epoch_{epoch}.png"))
        plt.close()