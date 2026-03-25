"""
evaluate.py  —  Blind denoising evaluation
-------------------------------------------
Reports:
  1. PSNR / SSIM / MAE  (denoising quality vs baseline)
  2. Sigma estimation MAE  (how accurately the model estimates noise level)
  3. PSNR breakdown by noise level  (low / medium / high sigma)
     — this is the most interesting analysis for a research write-up
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn
from pathlib import Path

from dataset import DenoisingDataset
from model   import UNet

CHECKPOINT = Path("checkpoints/best_model.pth")
BATCH_SIZE = 8

def tensor_to_np(t):
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet().to(device)
ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

# ── Test set ──────────────────────────────────────────────────────────────────
test_ds     = DenoisingDataset(split="test")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Test images: {len(test_ds)}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = []   # list of dicts, one per image

with torch.no_grad():
    for noisy_batch, clean_batch, sigma_batch in test_loader:
        noisy_batch  = noisy_batch.to(device)
        true_sigmas  = sigma_batch.squeeze(-1).squeeze(-1).cpu().numpy()   # (B,)

        pred_batch, pred_sigma_batch = model(noisy_batch)
        pred_batch       = pred_batch.cpu()
        pred_sigma_batch = pred_sigma_batch.squeeze(-1).cpu().numpy()       # (B,)

        for i in range(noisy_batch.size(0)):
            noisy_np = tensor_to_np(noisy_batch[i].cpu())
            clean_np = tensor_to_np(clean_batch[i])
            pred_np  = tensor_to_np(pred_batch[i])

            p = psnr_fn(clean_np, pred_np,  data_range=255)
            s = ssim_fn(clean_np, pred_np,  data_range=255, channel_axis=2)
            m = np.mean(np.abs(clean_np.astype(np.float32) - pred_np.astype(np.float32)))
            bp = psnr_fn(clean_np, noisy_np, data_range=255)

            results.append({
                "psnr": p, "ssim": s, "mae": m,
                "baseline_psnr": bp,
                "true_sigma": float(true_sigmas[i]),
                "pred_sigma": float(pred_sigma_batch[i]),
            })

# ── Overall results ───────────────────────────────────────────────────────────
print("\n" + "="*58)
print(f"{'Metric':<12}  {'Baseline':>14}  {'U-Net (blind)':>14}")
print("-"*58)
print(f"{'PSNR (dB)':<12}  {np.mean([r['baseline_psnr'] for r in results]):>14.2f}  {np.mean([r['psnr'] for r in results]):>14.2f}")
print(f"{'SSIM':<12}  {'—':>14}  {np.mean([r['ssim'] for r in results]):>14.4f}")
print(f"{'MAE':<12}  {'—':>14}  {np.mean([r['mae'] for r in results]):>14.2f}")
print(f"{'Sigma MAE':<12}  {'—':>14}  {np.mean([abs(r['true_sigma']-r['pred_sigma']) for r in results]):>14.2f}")
print("="*58)

# ── Breakdown by noise level ──────────────────────────────────────────────────
# This is the most interesting part — does the model do better on light noise?
print("\nPSNR breakdown by noise level:")
print(f"{'Sigma range':<18}  {'N images':>8}  {'PSNR':>8}")
print("-"*38)
for lo, hi in [(10, 30), (30, 55), (55, 75)]:
    subset = [r for r in results if lo <= r["true_sigma"] < hi]
    if subset:
        avg_psnr = np.mean([r["psnr"] for r in subset])
        label    = f"σ ∈ [{lo}, {hi})"
        print(f"{label:<18}  {len(subset):>8}  {avg_psnr:>8.2f}")

print("\nHigher PSNR & SSIM = better  |  Lower MAE & Sigma MAE = better")
print("\nNext step: python predict.py")