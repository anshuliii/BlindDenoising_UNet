"""
model.py  —  Blind U-Net with noise estimation head
----------------------------------------------------
Two outputs:
  1. clean  : (B, 3, H, W)  — denoised image  [0, 1]
  2. sigma  : (B, 1)        — estimated noise std  (0-255 scale)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            skip = skip[:, :, : x.shape[2], : x.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Blind denoising U-Net.
    Returns (clean_image, sigma_estimate).

    sigma_estimate is in the 0-255 pixel scale so it can be
    directly compared to the true noise std used during training.
    """

    def __init__(self, in_channels=3, out_channels=3, base_features=32):
        super().__init__()
        f = base_features

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, f)       # 3  → 32
        self.enc2 = Down(f,     f * 2)               # 32 → 64
        self.enc3 = Down(f * 2, f * 4)               # 64 → 128
        self.enc4 = Down(f * 4, f * 8)               # 128→ 256

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = Down(f * 8, f * 16)        # 256→ 512

        # ── Decoder ──────────────────────────────────────────────
        self.dec4 = Up(f * 16, f * 8,  f * 8)
        self.dec3 = Up(f * 8,  f * 4,  f * 4)
        self.dec2 = Up(f * 4,  f * 2,  f * 2)
        self.dec1 = Up(f * 2,  f,      f)

        # ── Output head 1: clean image ───────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # ── Output head 2: sigma estimation ──────────────────────
        # Takes the bottleneck features (richest representation),
        # global-average-pools them to a single vector, then
        # regresses to a single scalar — the estimated noise std.
        self.noise_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (B, 512, 1, 1)
            nn.Flatten(),                # (B, 512)
            nn.Linear(f * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(),                   # sigma is always >= 0
        )

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b  = self.bottleneck(s4)

        # Sigma estimation — from bottleneck (before decoding)
        sigma = self.noise_head(b)       # (B, 1)

        # Decoder
        x = self.dec4(b,  s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)

        clean = self.out_conv(x)         # (B, 3, H, W)

        return clean, sigma              # TWO outputs now


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model        = UNet()
    dummy        = torch.randn(2, 3, 128, 128)
    clean, sigma = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Clean shape  : {clean.shape}")
    print(f"Sigma shape  : {sigma.shape}")
    print(f"Sigma values : {sigma.detach().squeeze().tolist()}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}")