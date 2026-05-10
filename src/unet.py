import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def pos_encoding(times, output_dim, device="cpu"):
    half_dim = output_dim // 2
    emb_scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = times.float().view(-1, 1) * emb.view(1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if output_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w).transpose(2, 3)

        attn = torch.matmul(q, k) * (head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CondUNet(nn.Module):
    """Conditional residual U-Net for DDPM/DDIM on 32x32 images."""

    def __init__(
        self,
        in_ch=3,
        time_embed_dim=100,
        num_labels=None,
        label_scale=0.3,
        base_channels=64,
        dropout=0.1,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.num_labels = num_labels
        self.label_scale = label_scale
        emb_dim = time_embed_dim * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4

        self.in_conv = nn.Conv2d(in_ch, ch1, 3, padding=1)

        self.down1_a = ResBlock(ch1, ch1, emb_dim, dropout)
        self.down1_b = ResBlock(ch1, ch1, emb_dim, dropout)
        self.downsample1 = Downsample(ch1)

        self.down2_a = ResBlock(ch1, ch2, emb_dim, dropout)
        self.down2_b = ResBlock(ch2, ch2, emb_dim, dropout)
        self.downsample2 = Downsample(ch2)

        self.down3_a = ResBlock(ch2, ch3, emb_dim, dropout)
        self.down3_b = ResBlock(ch3, ch3, emb_dim, dropout)
        self.attn8 = AttentionBlock(ch3)
        self.downsample3 = Downsample(ch3)

        self.mid_a = ResBlock(ch3, ch3, emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch3)
        self.mid_b = ResBlock(ch3, ch3, emb_dim, dropout)

        self.upsample3 = Upsample(ch3)
        self.up3_a = ResBlock(ch3 + ch3, ch3, emb_dim, dropout)
        self.up3_b = ResBlock(ch3, ch3, emb_dim, dropout)
        self.up_attn8 = AttentionBlock(ch3)

        self.upsample2 = Upsample(ch3)
        self.up2_a = ResBlock(ch3 + ch2, ch2, emb_dim, dropout)
        self.up2_b = ResBlock(ch2, ch2, emb_dim, dropout)

        self.upsample1 = Upsample(ch2)
        self.up1_a = ResBlock(ch2 + ch1, ch1, emb_dim, dropout)
        self.up1_b = ResBlock(ch1, ch1, emb_dim, dropout)

        self.out_norm = _group_norm(ch1)
        self.out_conv = nn.Conv2d(ch1, in_ch, 3, padding=1)

    def forward(self, x, timesteps, labels=None):
        emb = pos_encoding(timesteps, self.time_embed_dim, x.device)
        if labels is not None:
            emb = emb + self.label_emb(labels) * self.label_scale
        emb = self.time_mlp(emb)

        x = self.in_conv(x)

        x = self.down1_a(x, emb)
        x1 = self.down1_b(x, emb)
        x = self.downsample1(x1)

        x = self.down2_a(x, emb)
        x2 = self.down2_b(x, emb)
        x = self.downsample2(x2)

        x = self.down3_a(x, emb)
        x = self.down3_b(x, emb)
        x3 = self.attn8(x)
        x = self.downsample3(x3)

        x = self.mid_a(x, emb)
        x = self.mid_attn(x)
        x = self.mid_b(x, emb)

        x = self.upsample3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3_a(x, emb)
        x = self.up3_b(x, emb)
        x = self.up_attn8(x)

        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2_a(x, emb)
        x = self.up2_b(x, emb)

        x = self.upsample1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1_a(x, emb)
        x = self.up1_b(x, emb)

        return self.out_conv(F.silu(self.out_norm(x)))


CondUnet = CondUNet
