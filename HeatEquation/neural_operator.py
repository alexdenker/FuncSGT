"""
1D Fourier Neural Operator with time-dependent embeddings for score-based modeling.

Adapted from: https://github.com/camlab-ethz/DSE-for-NeuralOperators/tree/main

"""

import torch
import torch.nn as nn
import math 


### Time embedding 
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

################################################################
# Fourier layer (using FFT)
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-freq modes to keep
        
        # Complex weights (real + imaginary parts)
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes, 2))
        self.weights_neg = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes, 2))
        
        # Time modulation scaling
        self.scale_A = 1 / emb_dim
        self.A_real_pos = nn.Parameter(
            self.scale_A * torch.randn(self.modes, emb_dim, dtype=torch.float))
        self.A_imag_pos = nn.Parameter(
            self.scale_A * torch.randn(self.modes, emb_dim, dtype=torch.float))        
        self.A_real_neg = nn.Parameter(
            self.scale_A * torch.randn(self.modes, emb_dim, dtype=torch.float))
        self.A_imag_neg = nn.Parameter(
            self.scale_A * torch.randn(self.modes, emb_dim, dtype=torch.float))    
    
    def compl_mul1d(self, input, weights):
        # input: (batch, in_channel, x), Fourier coeffs (complex)
        # weights: (in_channel, out_channel, x)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)
    
    def forward(self, x, emb):
        """
        x: (B, C_in, L) where L is sequence length
        emb: (B, emb_dim)
       
        Returns: (B, C_out, L)
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, norm="ortho")  # (B, C_in, L//2+1)
        
        # Build time modulation
        # emb: (B, emb_dim)
        # A_*: (modes, emb_dim)
        # Result should be: (B, modes)
        
        phi_real_pos = torch.einsum('mc,bc->bm', self.A_real_pos, emb)
        phi_imag_pos = torch.einsum('mc,bc->bm', self.A_imag_pos, emb)
        phi_complex_pos = phi_real_pos + 1j * phi_imag_pos  # (B, modes)
        
        phi_real_neg = torch.einsum('mc,bc->bm', self.A_real_neg, emb)
        phi_imag_neg = torch.einsum('mc,bc->bm', self.A_imag_neg, emb)
        phi_complex_neg = phi_real_neg + 1j * phi_imag_neg  # (B, modes)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply spectral convolution and time modulation to positive frequencies
        out_ft[:, :, :self.modes] = (
            self.compl_mul1d(
                x_ft[:, :, :self.modes],
                self.weights_pos
            ) * phi_complex_pos.unsqueeze(1)  # unsqueeze for channel dimension
        )
        
        # Apply spectral convolution and time modulation to negative frequencies
        out_ft[:, :, -self.modes:] = (
            self.compl_mul1d(
                x_ft[:, :, -self.modes:],
                self.weights_neg
            ) * phi_complex_neg.unsqueeze(1)  # unsqueeze for channel dimension
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), norm="ortho")
        return x

class TimeMLP(nn.Module):
    def __init__(self, emb_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),                  # or ReLU if you like pain
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, t):
        return self.net(t)   # [batch, out_dim]

class Conv1DTime(nn.Module):
    def __init__(self, width, emb_dim):
        super().__init__()

        self.time_mlp = TimeMLP(emb_dim=emb_dim, out_dim=2*width)
        self.w = nn.Linear(width, width)

    def forward(self, x, emb):
        """
        x: [batch_size, width, length]
        emb: [batch_size, emb_dim] timestep embedding
        
        return: [batch_size, width, length]
        """
        B, C, L = x.shape

        gamma, beta = self.time_mlp(emb).chunk(2, dim=-1)  # [batch, width] each
        
        # Apply MLP independently to each position: [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)  # [B, L, C]
        x = self.w(x)  # [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        # Apply time modulation
        x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return x

class FNOBlock(nn.Module):
    def __init__(self, modes, width, emb_dim):
        super(FNOBlock, self).__init__()

        self.modes = modes
        self.width = width
        self.emb_dim = emb_dim

        self.conv = SpectralConv1d(self.width, self.width, self.modes, self.emb_dim)
        self.w = Conv1DTime(self.width, self.emb_dim)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        """
        x: [batch_size, width, length] 
        emb: [batch_size, emb_dim] timestep embedding
        """

        x1 = self.conv(x, emb)
        x2 = self.w(x, emb)
        x = x1 + x2
        x = self.act(x)

        return x 
    
    
class FNO(nn.Module):
    def __init__(self, modes, width, n_layers, timestep_embedding_dim=33, max_period=2.0):
        super(FNO, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.max_period = max_period
        self.modes = modes 
        self.width = width
        self.n_layers = n_layers 

        self.fc0 = nn.Linear(2, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.SiLU()

        self.layers = nn.ModuleList()   
        for _ in range(self.n_layers):
            self.layers.append(FNOBlock(self.modes, self.width, self.timestep_embedding_dim))


    def forward(self, x, pos, t):
        """
        x: [batch, 1, num_points] - 1D signal
        pos: [batch, 1, num_points] - position encoding
        t: [batch] - timesteps
        
        Returns: [batch, num_points]
        """
        # Concatenate x and pos as two channels: [batch, 2, num_points]
        x = torch.cat([x, pos], dim=1)  # [batch, 2, num_points]
        
        # Get time embedding
        emb = timestep_embedding(t, self.timestep_embedding_dim, max_period=self.max_period)

        # Permute to [batch, num_points, 2] for linear layer
        x = x.permute(0, 2, 1)  # [batch, num_points, 2]
        x = self.fc0(x)  # [batch, num_points, width]
        x = x.permute(0, 2, 1)  # [batch, width, num_points]

        # Apply FNO layers
        for layer in self.layers:
            x = layer(x, emb)

        # Final projection
        x = x.permute(0, 2, 1)  # [batch, num_points, width]
        x = self.fc1(x)  # [batch, num_points, 128]
        x = self.act(x)
        x = self.fc2(x)  # [batch, num_points, 1]

        return x.squeeze(-1)  # [batch, num_points]


class ConditionalFNO(nn.Module):
    def __init__(self, modes, width, n_layers, timestep_embedding_dim=33, max_period=2.0):
        super(ConditionalFNO, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.max_period = max_period
        self.modes = modes 
        self.width = width
        self.n_layers = n_layers 

        self.fc0 = nn.Linear(3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.SiLU()

        self.layers = nn.ModuleList()   
        for _ in range(self.n_layers):
            self.layers.append(FNOBlock(self.modes, self.width, self.timestep_embedding_dim))


    def forward(self, x, y, pos, t):
        """
        x: [batch, 1, num_points] - 1D signal
        y: [batch, 1, num_points] - 1D signal
        pos: [batch, 1, num_points] - position encoding
        t: [batch] - timesteps
        
        Returns: [batch, num_points]
        """
        # Concatenate x and pos as two channels: [batch, 2, num_points]
        x = torch.cat([x, y, pos], dim=1)  # [batch, 3, num_points]

        # Get time embedding
        emb = timestep_embedding(t, self.timestep_embedding_dim, max_period=self.max_period)

        # Permute to [batch, num_points, 2] for linear layer
        x = x.permute(0, 2, 1)  # [batch, num_points, 2]
        x = self.fc0(x)  # [batch, num_points, width]
        x = x.permute(0, 2, 1)  # [batch, width, num_points]

        # Apply FNO layers
        for layer in self.layers:
            x = layer(x, emb)

        # Final projection
        x = x.permute(0, 2, 1)  # [batch, num_points, width]
        x = self.fc1(x)  # [batch, num_points, 128]
        x = self.act(x)
        x = self.fc2(x)  # [batch, num_points, 1]

        return x.squeeze(-1)  # [batch, num_points]



class CondFNO(nn.Module):
    def __init__(self, modes, width, n_layers, timestep_embedding_dim=33, max_period=2.0):
        super(CondFNO, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.max_period = max_period
        self.modes = modes 
        self.width = width
        self.n_layers = n_layers 

        self.fc0 = nn.Linear(3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.SiLU()

        self.layers = nn.ModuleList()   
        for _ in range(self.n_layers):
            self.layers.append(FNOBlock(self.modes, self.width, self.timestep_embedding_dim))

        self.log_likelihood_scaling = nn.Sequential(
            nn.Linear(self.timestep_embedding_dim, self.timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.timestep_embedding_dim, self.timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.timestep_embedding_dim, 1),
        )

        self.log_likelihood_scaling[-1].weight.data.fill_(0.0)
        self.log_likelihood_scaling[-1].bias.data.fill_(0.01)

    def forward(self, x, y, log_grad, pos, t):
        """
        x: [batch, 1, num_points] - 1D signal
        y: [batch, 1, num_points] - 1D signal
        log_grad: [batch, num_points] - log likelihood gradient w.r.t. y
        pos: [batch, 1, num_points] - position encoding
        t: [batch] - timesteps

        Returns: [batch, num_points]
        """
        # Concatenate x and pos as two channels: [batch, 2, num_points]
        x = torch.cat([x, y.unsqueeze(1), pos], dim=1)  # [batch, 2, num_points]
        
        # Get time embedding
        emb = timestep_embedding(t, self.timestep_embedding_dim, max_period=self.max_period)

        # Permute to [batch, num_points, 2] for linear layer
        x = x.permute(0, 2, 1)  # [batch, num_points, 2]
        x = self.fc0(x)  # [batch, num_points, width]
        x = x.permute(0, 2, 1)  # [batch, width, num_points]

        # Apply FNO layers
        for layer in self.layers:
            x = layer(x, emb)

        # Final projection
        x = x.permute(0, 2, 1)  # [batch, num_points, width]
        x = self.fc1(x)  # [batch, num_points, 128]
        x = self.act(x)
        x = self.fc2(x)  # [batch, num_points, 1]

        # Scale output by log likelihood gradient
        scaling = self.log_likelihood_scaling(emb)  # [batch, 1]
        out = x.squeeze(-1) + scaling * log_grad.squeeze(1)  # [batch, input_dim]
        return out  # [batch, num_points]

if __name__ == "__main__":
    # test 1D FNO 
    
    model = FNO(modes=16, width=32, n_layers=4, timestep_embedding_dim=33)
    x = torch.randn(4, 256)  # [batch=4, num_points=256]
    t = torch.randint(0, 1, (4,))
    y = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} != {x.shape}"

    