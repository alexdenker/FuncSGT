import math

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


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
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * out_dim))
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Skip connection: handle dimension mismatch
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, t_emb):
        # Apply first layer with normalization
        h = self.norm1(self.fc1(x))
        h = F.silu(h)

        # Apply time conditioning
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)  # [batch, out_dim] each
        h = h * (1 + scale) + shift

        # Second layer with normalization
        h = self.norm2(self.fc2(h))
        h = F.silu(h)

        # Skip connection
        skip = self.skip(x)
        return h + skip


class ScoreNet(nn.Module):
    def __init__(
        self,
        input_dim=64,
        output_dim=64,
        hidden_dim=256,
        time_embed_dim=64,
        depth=6,
        max_period=2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.max_period = max_period

        # Project input to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_layers = nn.ModuleList()  # Track layer dimensions

        current_dim = hidden_dim
        for i in range(depth):
            self.down_blocks.append(
                ResidualBlock(current_dim, hidden_dim, time_embed_dim)
            )
            current_dim = hidden_dim

        # Bottleneck
        self.mid = ResidualBlock(hidden_dim, hidden_dim, time_embed_dim)

        # Decoder (upsampling path with skip connections)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            # Skip connections: concatenate with corresponding encoder output
            # So input is 2*hidden_dim (hidden + skip)
            self.up_blocks.append(
                ResidualBlock(2 * hidden_dim, hidden_dim, time_embed_dim)
            )

        # Project back to input dimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize output layer to near-zero (important for score matching)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, t):
        """
        Args:
            x: Input tensor of shape [batch, input_dim]
            t: Time tensor of shape [batch]

        Returns:
            score: Score function of shape [batch, input_dim]
        """
        # Embed time
        t_emb = timestep_embedding(
            t, dim=self.time_embed_dim, max_period=self.max_period
        )
        t_emb = self.time_mlp(t_emb)  # [batch, time_embed_dim]

        # Project input to hidden dimension
        h = self.input_layer(x)  # [batch, hidden_dim]

        # Encoder: store skip connections
        skips = []
        for down in self.down_blocks:
            skips.append(h)
            h = down(h, t_emb)

        # Bottleneck
        h = self.mid(h, t_emb)

        # Decoder: use skip connections
        for up in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=-1)  # [batch, 2*hidden_dim]
            h = up(h, t_emb)

        # Project back to input dimension with residual connection
        out = self.output_layer(h)  # [batch, input_dim]

        return out


class CondScoreNet(nn.Module):
    def __init__(
        self,
        input_dim=64,
        output_dim=64,
        hidden_dim=256,
        time_embed_dim=64,
        depth=6,
        max_period=2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.max_period = max_period

        # Project input to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_layers = nn.ModuleList()  # Track layer dimensions

        current_dim = hidden_dim
        for i in range(depth):
            self.down_blocks.append(
                ResidualBlock(current_dim, hidden_dim, time_embed_dim)
            )
            current_dim = hidden_dim

        # Bottleneck
        self.mid = ResidualBlock(hidden_dim, hidden_dim, time_embed_dim)

        # Decoder (upsampling path with skip connections)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            # Skip connections: concatenate with corresponding encoder output
            # So input is 2*hidden_dim (hidden + skip)
            self.up_blocks.append(
                ResidualBlock(2 * hidden_dim, hidden_dim, time_embed_dim)
            )

        # Project back to input dimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize output layer to near-zero (important for score matching)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.log_likelihood_scaling = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, 1),
        )

        self.log_likelihood_scaling[-1].weight.data.fill_(0.0)
        self.log_likelihood_scaling[-1].bias.data.fill_(0.01)

    def forward(self, x, log_grad, t):
        """
        Args:
            x: Input tensor of shape [batch, input_dim]
            t: Time tensor of shape [batch]

        Returns:
            score: Score function of shape [batch, input_dim]
        """
        # Embed time
        t_embedding = timestep_embedding(
            t, dim=self.time_embed_dim, max_period=self.max_period
        )
        t_emb = self.time_mlp(t_embedding)  # [batch, time_embed_dim]

        # Project input to hidden dimension
        h = self.input_layer(x)  # [batch, hidden_dim]

        # Encoder: store skip connections
        skips = []
        for down in self.down_blocks:
            skips.append(h)
            h = down(h, t_emb)

        # Bottleneck
        h = self.mid(h, t_emb)

        # Decoder: use skip connections
        for up in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=-1)  # [batch, 2*hidden_dim]
            h = up(h, t_emb)

        # Project back to input dimension with residual connection
        out = self.output_layer(h)  # [batch, input_dim]

        # Scale output by log likelihood gradient
        scaling = self.log_likelihood_scaling(t_embedding)  # [batch, 1]

        out = out + scaling * log_grad  # [batch, input_dim]

        return out
