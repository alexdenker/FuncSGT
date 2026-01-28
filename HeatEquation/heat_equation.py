import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class HeatEquation1D(nn.Module):
    """
    Differentiable 1D Heat Equation Solver using Crank-Nicolson method.
    Domain: x in [0, 1]
    BCs: u(0, t) = u(1, t) = 0 (Zero Boundary Conditions)
    """

    def __init__(self, nx=64, nu=0.01, dt=0.001, t_max=0.1, device="cpu"):
        super().__init__()
        self.nx = nx  # Number of grid points
        self.nu = nu  # Viscosity / Diffusion coefficient
        self.dt = dt  # Time step size
        self.t_max = t_max  # Final time
        self.device = device

        # Spatial discretization
        self.dx = 1.0 / (nx - 1)
        self.x_grid = torch.linspace(0, 1, nx).to(device)

        # Precompute the Crank-Nicolson matrices
        # We solve: (I - 0.5 * alpha * L) u_{t+1} = (I + 0.5 * alpha * L) u_t
        # Where L is the Laplacian matrix and alpha = nu * dt / dx^2

        alpha = (self.nu * self.dt) / (self.dx**2)

        # Construct Tridiagonal Laplacian Matrix (discrete 2nd derivative)
        # Diagonals: -2, Off-diagonals: 1
        main_diag = -2 * torch.ones(nx - 2, device=device)
        off_diag = torch.ones(nx - 3, device=device)

        L = (
            torch.diag(main_diag)
            + torch.diag(off_diag, diagonal=1)
            + torch.diag(off_diag, diagonal=-1)
        )

        # Crank-Nicolson matrices (for internal points only to enforce BCs)
        # A_lhs * u_{t+1} = A_rhs * u_t
        I = torch.eye(nx - 2, device=device)
        self.A_lhs = I - 0.5 * alpha * L
        self.A_rhs = I + 0.5 * alpha * L

        # Precompute the inverse of LHS for fast solving
        # u_{t+1} = (A_lhs^-1 @ A_rhs) u_t = Step_Matrix @ u_t
        self.step_matrix = torch.linalg.solve(self.A_lhs, self.A_rhs)

        # Calculate number of steps
        self.num_steps = int(t_max / dt)

    def forward(self, u0):
        """
        Args:
            u0: Initial condition tensor of shape (Batch, nx)
        Returns:
            u_t: Solution at t_max of shape (Batch, nx)
        """
        # Enforce zero BCs strictly by only operating on inner points
        # u0 shape: [Batch, nx]
        u_inner = u0[:, 1:-1]  # Discard boundary points (assumed 0 or ignored)

        # Time stepping
        # Since the system is linear, we can power the matrix if memory allows,
        # but iterative is safer for large grids/autograd.
        # Ideally, for fixed T, we could precompute step_matrix^num_steps.

        # Option A: Iterative (good if you need intermediate states)
        curr_u = u_inner.T  # Transpose to [nx-2, Batch] for matmul
        for _ in range(self.num_steps):
            curr_u = torch.matmul(self.step_matrix, curr_u)

        # Option B (Faster for fixed T): Precompute matrix power in __init__
        # global_operator = torch.linalg.matrix_power(self.step_matrix, self.num_steps)
        # curr_u = torch.matmul(global_operator, u_inner.T)

        curr_u = curr_u.T  # Back to [Batch, nx-2]

        # Pad with zeros for boundary conditions
        batch_size = u0.shape[0]
        zeros = torch.zeros(batch_size, 1, device=self.device)
        u_final = torch.cat([zeros, curr_u, zeros], dim=1)

        return u_final


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = HeatEquation1D(nx=100, nu=0.05, dt=0.001, t_max=0.05, device=device)

    # 2. Create a smooth initial condition (e.g., a Gaussian or Sine wave)
    # u0(x) = sin(pi * x) + 0.5 * sin(3 * pi * x)
    x = solver.x_grid.unsqueeze(0)  # [1, nx]
    # u0 = torch.sin(math.pi * x) + 0.5 * torch.sin(3 * math.pi * x)

    lamb = np.random.rand() * 0.3 + 0.2

    u0 = torch.exp(-100 * (x - lamb) ** 2) - torch.exp(-100 * (x - (1 - lamb)) ** 2)

    print("Initial condition u0 shape:", u0.shape)
    # 3. Solve Forward (Diffusing the initial condition)
    with torch.no_grad():
        u_T = solver(u0)

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(
        x.cpu().numpy().flatten(),
        u0.cpu().numpy().flatten(),
        label="Initial Condition ($u_0$)",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        x.cpu().numpy().flatten(),
        u_T.cpu().numpy().flatten(),
        label=f"Diffused State ($u_T, T={solver.t_max}$)",
        linewidth=2,
    )
    plt.title(f"1D Heat Equation (Forward Pass)\n$\\nu={solver.nu}, T={solver.t_max}$")
    plt.xlabel("Domain $x$")
    plt.ylabel("u(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("1D_heat_equation_forward.png", dpi=150)
    plt.show()

    ### try to solve inverse problem

    # B. Generate Observation y (Forward Pass + Noise)
    with torch.no_grad():
        sigma = 0.05  # 5% noise
        noise = torch.randn_like(u_T) * sigma
        y_obs = u_T + noise

    u0_hat = torch.zeros_like(u0, requires_grad=True, device=device)

    # Optimizer
    # We use Adam. LBFGS is also common for physics, but Adam is standard for DL.
    optimizer = torch.optim.Adam([u0_hat], lr=0.01)

    # Optimization Loop
    losses = []
    for step in range(2000):
        optimizer.zero_grad()

        # 1. Forward pass (Prediction)
        y_pred = solver(u0_hat)

        # 2. Compute Loss (Data Fidelity)
        # min || A(u) - y ||^2
        loss = nn.MSELoss()(y_pred, y_obs)

        # Optional: Tikhonov Regularization (Ridge Regression)
        loss += 0.001 * torch.norm(u0_hat)

        # 3. Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 200 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")

    # --- 4. Visualization ---
    plt.figure(figsize=(14, 5))

    # Plot 1: The Problem Setup
    plt.subplot(1, 2, 1)
    plt.plot(
        x.cpu().flatten(),
        u0.cpu().flatten(),
        "k--",
        linewidth=2,
        label="True Initial $u_0$",
    )
    plt.plot(
        x.cpu().flatten(),
        y_obs.cpu().flatten(),
        "r",
        alpha=0.6,
        label="Noisy Observation $y$ (at T)",
    )
    plt.plot(
        x.cpu().flatten(),
        u_T.cpu().flatten(),
        "g:",
        alpha=0.8,
        label="Clean Signal (at T)",
    )
    plt.title("Problem Setup: Forward Diffusion + Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: The Failed Reconstruction
    plt.subplot(1, 2, 2)
    plt.plot(
        x.cpu().flatten(),
        u0.cpu().flatten(),
        "k--",
        linewidth=2,
        label="True Initial $u_0$",
    )
    plt.plot(
        x.cpu().flatten(),
        u0_hat.detach().cpu().flatten(),
        "b-",
        linewidth=1,
        label="Classical Solution $\hat{u}_0$",
    )
    plt.title(f"Classical Inverse Result\n(Minimizing MSE, {len(losses)} steps)")
    plt.legend()
    plt.ylim(-1, 1)  # Limit y-axis because the noise blows up
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
