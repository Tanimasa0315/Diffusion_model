import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim=100):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x):
        return self.convs(x)

    def forward_time_embed(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class SimpleUnet(nn.Module):
    """U-Netの簡易版
    """

    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)

        x = self.bot1(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x


class SimpleUnetWithTime(nn.Module):
    """位置情報(時刻t)を埋め込むU-Net Model"""

    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1.forward_time_embed(x, v)
        x = self.maxpool(x1)
        x2 = self.down2.forward_time_embed(x, v)
        x = self.maxpool(x2)

        x = self.bot1.forward_time_embed(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2.forward_time_embed(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1.forward_time_embed(x, v)
        x = self.out(x)
        return x


class CondSimpleUnet(nn.Module):
    """位置情報(時刻t)および条件yを埋め込むU-Net Model"""

    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None, label_scale=0.3):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.label_scale = label_scale

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        # 正弦波位置エンコーディングによる時間の埋め込み
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        # 条件の埋め込み
        if labels is not None:
            v += self.label_emb(labels) * self.label_scale

        x1 = self.down1.forward_time_embed(x, v)
        x = self.maxpool(x1)
        x2 = self.down2.forward_time_embed(x, v)
        x = self.maxpool(x2)

        x = self.bot1.forward_time_embed(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2.forward_time_embed(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1.forward_time_embed(x, v)
        x = self.out(x)
        return x


class CondSimpleUnetDeep(nn.Module):
    """CondSimpleUnetを一段深くしたU-Net Model"""

    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None, label_scale=0.3):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.num_labels = num_labels
        self.label_scale = label_scale

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.down3 = ConvBlock(128, 256, time_embed_dim)
        self.bot1 = ConvBlock(256, 512, time_embed_dim)
        self.up3 = ConvBlock(512 + 256, 256, time_embed_dim)
        self.up2 = ConvBlock(256 + 128, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        if labels is not None:
            v += self.label_emb(labels) * self.label_scale

        x1 = self.down1.forward_time_embed(x, v)   # [B, 64, 32, 32]
        x = self.maxpool(x1)                       # [B, 64, 16, 16]
        x2 = self.down2.forward_time_embed(x, v)   # [B, 128, 16, 16]
        x = self.maxpool(x2)                       # [B, 128, 8, 8]
        x3 = self.down3.forward_time_embed(x, v)   # [B, 256, 8, 8]
        x = self.maxpool(x3)                       # [B, 256, 4, 4]

        x = self.bot1.forward_time_embed(x, v)     # [B, 512, 4, 4]

        x = self.upsample(x)                       # [B, 512, 8, 8]
        x = torch.cat([x, x3], dim=1)              # [B, 768, 8, 8]
        x = self.up3.forward_time_embed(x, v)      # [B, 256, 8, 8]
        x = self.upsample(x)                       # [B, 256, 16, 16]
        x = torch.cat([x, x2], dim=1)              # [B, 384, 16, 16]
        x = self.up2.forward_time_embed(x, v)      # [B, 128, 16, 16]
        x = self.upsample(x)                       # [B, 128, 32, 32]
        x = torch.cat([x, x1], dim=1)              # [B, 192, 32, 32]
        x = self.up1.forward_time_embed(x, v)      # [B, 64, 32, 32]

        x = self.out(x)
        return x


# 正弦波位置エンコーディング
def _pos_encoding(time, output_dim, device='cpu'):
    v = torch.zeros(output_dim, device=device)

    i = torch.arange(0, output_dim, device=device)
    div_term = 10000 ** (i/output_dim)

    v[0::2] = torch.sin(time / div_term[0::2])
    v[1::2] = torch.cos(time / div_term[1::2])
    return v


# バッチ処理に対応
def pos_encoding(times, output_dim, device='cpu'):
    batch_size = len(times)
    v = torch.zeros(batch_size, output_dim, device=device)

    for i in range(batch_size):
        v[i] = _pos_encoding(times[i], output_dim, device)
    return v
