import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class PointEvaluationOperator(nn.Module):
    """
    Evaluates a function at specific points using integration with a narrow Gaussian.
    Domain: x in [0, 1]

    Given function values on a grid, this operator:
    1. Interpolates the function using the grid values
    2. Evaluates at specific measurement points using Gaussian weighting
    """

    def __init__(self, nx=64, n_points=10, sigma=0.01, seed=None, device="cpu"):
        super().__init__()
        self.nx = nx  # Number of grid points for the function
        self.n_points = n_points  # Number of measurement points
        self.sigma = sigma  # Width of the Gaussian for integration
        self.device = device

        # Spatial discretization
        self.dx = 1.0 / (nx - 1)
        self.x_grid = torch.linspace(0, 1, nx).to(device)

        # Measurement points (randomly distributed)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        self.eval_points = torch.rand(
            n_points, device=device, generator=generator
        ).sort()[0]

    def forward(self, u):
        """
        Args:
            u: Function values on grid, shape (Batch, nx)
        Returns:
            measurements: Evaluated function at measurement points, shape (Batch, n_points)
        """
        measurements = []

        # For each measurement point, integrate with narrow Gaussian
        for eval_point in self.eval_points:
            # Compute Gaussian weights centered at eval_point
            # w(x) = exp(-(x - eval_point)^2 / (2 * sigma^2))
            weights = torch.exp(-0.5 * ((self.x_grid - eval_point) / self.sigma) ** 2)

            # Normalize weights to act as an approximate delta function
            weights = weights / (weights.sum() + 1e-8)

            # Weighted sum: integral of u(x) * w(x) dx
            measurement = torch.sum(u * weights.unsqueeze(0), dim=1, keepdim=True)
            measurements.append(measurement)

        # Stack measurements along feature dimension
        measurements = torch.cat(measurements, dim=1)

        return measurements


class Conditioning(nn.Module):
    """
    Lifts point observations (x_i, y_i) to grid-aligned conditioning
    channels using Gaussian kernels.

    Produces:
        v(x): kernel-smoothed values
        m(x): kernel mass (mask / confidence)
    """

    def __init__(self, nx=64, sigma=0.02, device="cpu"):
        super().__init__()
        self.nx = nx
        self.sigma = sigma
        self.device = device

        self.x_grid = torch.linspace(0, 1, nx, device=device)  # (nx,)

    def forward(self, eval_points, values):
        """
        Args:
            eval_points: (Batch, k)  locations in [0,1]
            values:      (Batch, k)  observed values f(x_i)

        Returns:
            v: (Batch, nx)  smoothed values
            m: (Batch, nx)  kernel mass (mask)
        """
        # print("eval_points:", eval_points.shape)
        # print("values:", values.shape)

        B, k = eval_points.shape
        nx = self.nx

        # Shape bookkeeping
        # x_grid:      (1, 1, nx)
        # eval_points: (B, k, 1)
        xg = self.x_grid.view(1, 1, nx)
        xp = eval_points.unsqueeze(-1)

        # Gaussian kernel
        # (B, k, nx)
        kernel = torch.exp(-0.5 * ((xg - xp) / self.sigma) ** 2)

        # Kernel mass (mask)
        # (B, nx)
        m = kernel.sum(dim=1)

        # Weighted values
        # (B, nx)
        v = (kernel * values.unsqueeze(-1)).sum(dim=1)

        return v, m


if __name__ == "__main__":
    from prior import PriorDataset

    # --- Example Usage ---
    # 1. Setup
    # n_points: 10
    # sigma: 0.02
    # seed: 42
    # noise_std: 0.01
    # cond_sigma: 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    operator = PointEvaluationOperator(
        nx=128, n_points=10, sigma=0.02, seed=42, device=device
    )

    # 2. Create a smooth initial condition (e.g., Gaussian bumps)
    x = operator.x_grid.unsqueeze(0)  # [1, nx]

    lamb = np.random.rand() * 0.3 + 0.2

    # Create a test function with two Gaussian bumps
    dataset = PriorDataset(n_samples=1, n_points=128)
    u0 = dataset[0].to(device)  # [1, nx]
    print("Initial condition u0 shape:", u0.shape)

    # 3. Evaluate at measurement points
    with torch.no_grad():
        measurements = operator(u0)

    print("Measurements shape:", measurements.shape)

    cond_operator = Conditioning(nx=128, sigma=0.01, device=device)
    v, m = cond_operator(operator.eval_points.unsqueeze(0), measurements)

    # 4. Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        x.cpu().numpy().flatten(),
        u0.cpu().numpy().flatten(),
        label="Function $u(x)$",
        linewidth=2,
    )
    plt.scatter(
        operator.eval_points.cpu().numpy(),
        measurements.cpu().numpy().flatten(),
        color="red",
        s=50,
        label="Measurement Points",
        zorder=5,
    )
    plt.plot(
        x.cpu().numpy().flatten(),
        v.cpu().numpy().flatten(),
        label="Conditioned $v(x)$",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        x.cpu().numpy().flatten(),
        m.cpu().numpy().flatten(),
        label="Mask $m(x)$",
        linestyle=":",
        linewidth=2,
    )
    plt.title("Point Evaluation with Gaussian Integration")
    plt.xlabel("Domain $x$")
    plt.ylabel("u(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(len(measurements.flatten())), measurements.cpu().numpy().flatten())
    plt.xlabel("Measurement Index")
    plt.ylabel("Measured Value")
    plt.title("Measurement Values at Evaluation Points")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("point_evaluation.png", dpi=150)
    plt.show()

    # --- Inverse Problem ---
    # B. Generate Observation y (Add noise to measurements)
    with torch.no_grad():
        sigma_noise = 0.01  # 1% noise
        noise = torch.randn_like(measurements) * sigma_noise
        y_obs = measurements + noise

    # Initial guess for the function
    u_hat = torch.zeros_like(u0, requires_grad=True, device=device)

    # Optimizer
    optimizer = torch.optim.Adam([u_hat], lr=0.01)

    # Optimization Loop
    losses = []
    for step in range(2000):
        optimizer.zero_grad()

        # 1. Forward pass (Prediction)
        y_pred = operator(u_hat)

        # 2. Compute Loss (Data Fidelity)
        loss = nn.MSELoss()(y_pred, y_obs)

        # Optional: Tikhonov Regularization
        loss += 0.001 * torch.norm(u_hat)

        # 3. Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 200 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")

    # --- 4. Visualization of Inverse Problem ---
    plt.figure(figsize=(14, 5))

    # Plot 1: The Problem Setup
    plt.subplot(1, 3, 1)
    plt.plot(
        x.cpu().flatten(),
        u0.cpu().flatten(),
        "k--",
        linewidth=2,
        label="True Function $u(x)$",
    )
    plt.scatter(
        operator.eval_points.cpu().numpy(),
        y_obs.cpu().numpy().flatten(),
        color="red",
        s=50,
        label="Noisy Measurements",
        zorder=5,
    )
    plt.scatter(
        operator.eval_points.cpu().numpy(),
        measurements.cpu().numpy().flatten(),
        color="green",
        s=30,
        label="Clean Measurements",
        marker="x",
        zorder=4,
    )
    plt.title("Inverse Problem Setup")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: The Reconstruction
    plt.subplot(1, 3, 2)
    plt.plot(
        x.cpu().flatten(),
        u0.cpu().flatten(),
        "k--",
        linewidth=2,
        label="True Function $u(x)$",
    )
    plt.plot(
        x.cpu().flatten(),
        u_hat.detach().cpu().flatten(),
        "b-",
        linewidth=2,
        label="Reconstructed $\hat{u}(x)$",
    )
    plt.scatter(
        operator.eval_points.cpu().numpy(),
        y_obs.cpu().numpy().flatten(),
        color="red",
        s=50,
        label="Noisy Measurements",
        zorder=5,
    )
    plt.title("Reconstructed Function")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Loss curve
    plt.subplot(1, 3, 3)
    plt.semilogy(losses)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Optimization Progress ({len(losses)} steps)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("point_evaluation_inverse.png", dpi=150)
    plt.show()
