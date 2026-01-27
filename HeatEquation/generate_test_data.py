import torch 
import numpy as np 
import yaml 

from prior import PriorDataset
from heat_equation import HeatEquation1D


with open("configs/forward_op.yaml", 'r') as f:
    forward_op_config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_points = 128
solver = HeatEquation1D(nx=n_points, 
                        nu=forward_op_config['nu'], 
                        dt=forward_op_config['dt'], 
                        t_max=forward_op_config['t_max'], 
                        device=device)



num_dataset_elements = 64


dataset = PriorDataset(n_samples=num_dataset_elements, n_points=n_points)

x_gt_list = [] 
y_noises_list = []
y_cleans_list = []
for dataset_idx in range(num_dataset_elements):
    print(f"\n[{dataset_idx+1}/{num_dataset_elements}] Processing dataset element {dataset_idx}")
    
    # Load ground truth
    x_gt = dataset[dataset_idx].to(device) # shape: [1, nx]
    print("x_gt:", x_gt.shape)
    with torch.no_grad():
        y_gt = solver(x_gt)
        sigma = forward_op_config['noise_std']
        torch.manual_seed(dataset_idx)
        y_noise =  y_gt + torch.randn_like(y_gt) * sigma

    print("y_noise:", y_noise.shape)
    x_gt_list.append(x_gt.cpu().numpy())
    y_noises_list.append(y_noise.cpu().numpy())
    y_cleans_list.append(y_gt.cpu().numpy())

x_gt_all = np.concatenate(x_gt_list, axis=0)
y_noise_al = np.concatenate(y_noises_list, axis=0)
y_clean_all = np.concatenate(y_cleans_list, axis=0)

print("x_gt_all shape:", x_gt_all.shape)
print("y_noise_all shape:", y_noise_al.shape)
print("y_clean_all shape:", y_clean_all.shape)

np.save("test_data/ground_truth.npy", x_gt_all)
np.save("test_data/noisy_observations.npy", y_noise_al)
np.save("test_data/clean_observations.npy", y_clean_all)