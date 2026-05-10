import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
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


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(emb_dim, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = h + self.time_proj(emb).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        emb_dim=128,
        max_period=10000,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_period = max_period

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_channels * m for m in channel_mults]

        in_ch = base_channels
        for idx, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1D(in_ch, ch, emb_dim))
                in_ch = ch
            self.down_blocks.append(blocks)
            if idx < len(channels) - 1:
                self.downsamples.append(Downsample1D(in_ch))

        self.mid_block1 = ResBlock1D(in_ch, in_ch, emb_dim)
        self.mid_block2 = ResBlock1D(in_ch, in_ch, emb_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx, ch in reversed(list(enumerate(channels))):
            if idx < len(channels) - 1:
                self.upsamples.append(Upsample1D(in_ch))
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1D(in_ch + ch, ch, emb_dim))
                in_ch = ch
            self.up_blocks.append(blocks)

        self.out_conv = nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [batch, 1, num_points]
        t: [batch]
        """
        emb = timestep_embedding(t, self.emb_dim, max_period=self.max_period)
        emb = self.time_mlp(emb)

        h = self.in_conv(x)
        skips = []

        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)

        upsample_idx = 0
        for idx, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if idx > 0:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, emb)

        out = self.out_conv(h)
        return out.squeeze(1)
