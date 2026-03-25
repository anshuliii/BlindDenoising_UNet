"""
Saves a 3-panel comparison:
    Noisy  |  Model Output  |  Ground Truth

Also prints the model's sigma estimate vs true sigma.

Usage:
    python predict.py                          # random test image
    python predict.py --img path/to/image.png  # your own image
"""

import argparse
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

from dataset import DenoisingDataset, NOISY_DIR, CLEAN_DIR, SPLITS_DIR, IMG_SIZE, NOISE_MIN, NOISE_MAX
from model   import UNet

CHECKPOINT = Path("checkpoints/best_model.pth")
OUT_PATH   = Path("outputs/prediction_comparison.png")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default=None)
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet().to(device)
ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']})")

to_tensor = transforms.ToTensor()

# ── Load image ────────────────────────────────────────────────────────────────
if args.img:
    img        = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB)
    clean      = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    true_sigma = random.uniform(NOISE_MIN, NOISE_MAX)
    noise      = np.random.normal(0, true_sigma, clean.shape).astype(np.float32)
    noisy      = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
else:
    with open(SPLITS_DIR / "test.txt") as f:
        filenames = [l.strip() for l in f]
    name  = random.choice(filenames)
    stem  = name.replace(".png", "")
    noisy = cv2.cvtColor(cv2.imread(str(NOISY_DIR / name)), cv2.COLOR_BGR2RGB)
    clean = cv2.cvtColor(cv2.imread(str(CLEAN_DIR / name)), cv2.COLOR_BGR2RGB)
    with open(Path("data/sigma") / f"{stem}.txt") as f:
        true_sigma = float(f.read().strip())

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    inp              = to_tensor(noisy).unsqueeze(0).to(device)
    pred_clean, pred_sigma = model(inp)
    pred_clean       = pred_clean.squeeze(0).cpu()
    pred_sigma_val   = pred_sigma.item()

pred_np = (pred_clean.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

# ── Metrics ───────────────────────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

psnr_noisy = psnr_fn(clean, noisy,   data_range=255)
psnr_pred  = psnr_fn(clean, pred_np, data_range=255)
ssim_pred  = ssim_fn(clean, pred_np, data_range=255, channel_axis=2)

print(f"True sigma  : {true_sigma:.1f}")
print(f"Pred sigma  : {pred_sigma_val:.1f}  (error: {abs(true_sigma - pred_sigma_val):.1f})")
print(f"Noisy PSNR  : {psnr_noisy:.2f} dB")
print(f"Model PSNR  : {psnr_pred:.2f} dB")
print(f"Model SSIM  : {ssim_pred:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img_np, title in zip(
    axes,
    [noisy, pred_np, clean],
    [f"Noisy (σ={true_sigma:.1f})\nPSNR={psnr_noisy:.2f} dB",
     f"U-Net output (est. σ={pred_sigma_val:.1f})\nPSNR={psnr_pred:.2f} dB",
     "Ground truth"],
):
    ax.imshow(img_np)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.suptitle("Blind Denoising Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\nSaved → {OUT_PATH}")