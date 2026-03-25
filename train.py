"""
train.py  —  Blind denoising training
--------------------------------------
Key differences from v1:
  - Dataset now returns (noisy, clean, sigma)
  - Model returns (pred_clean, pred_sigma)
  - Loss = denoising_loss + SIGMA_WEIGHT * sigma_loss
  - Logs sigma MAE separately so you can track estimation accuracy
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import DenoisingDataset
from model   import UNet

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 8
LR           = 1e-3
NUM_WORKERS  = 0

# How much to weight the sigma estimation loss vs the denoising loss.
# 0.01 means: "care about denoising 100x more, but still learn sigma."
# Increase to 0.1 if you want stronger sigma estimation.
SIGMA_WEIGHT = 0.01

CHECKPOINT   = Path("checkpoints/best_model.pth")
PLOT_PATH    = Path("outputs/loss_curve.png")

# ── Setup ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
train_ds = DenoisingDataset(split="train")
val_ds   = DenoisingDataset(split="val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

# ── Model, losses, optimiser ──────────────────────────────────────────────────
model          = UNet().to(device)
denoise_loss   = nn.MSELoss()       # for pixel reconstruction
sigma_loss_fn  = nn.MSELoss()       # for sigma estimation
optimizer      = optim.Adam(model.parameters(), lr=LR)
scheduler      = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    running_loss = 0.0

    for noisy, clean, true_sigma in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        true_sigma   = true_sigma.squeeze(-1).to(device)   # (B, 1)

        optimizer.zero_grad()

        pred_clean, pred_sigma = model(noisy)

        # Combined loss: reconstruction + sigma estimation
        loss_d = denoise_loss(pred_clean, clean)
        loss_s = sigma_loss_fn(pred_sigma, true_sigma)
        loss   = loss_d + SIGMA_WEIGHT * loss_s

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    train_loss = running_loss / len(train_ds)

    # ── Validate ──
    model.eval()
    val_loss     = 0.0
    sigma_errors = []      # track how well sigma is estimated

    with torch.no_grad():
        for noisy, clean, true_sigma in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            true_sigma   = true_sigma.squeeze(-1).to(device)

            pred_clean, pred_sigma = model(noisy)

            loss_d = denoise_loss(pred_clean, clean)
            loss_s = sigma_loss_fn(pred_sigma, true_sigma)
            loss   = loss_d + SIGMA_WEIGHT * loss_s

            val_loss += loss.item() * noisy.size(0)

            # Sigma MAE: how close is the estimate to true sigma?
            sigma_mae = torch.abs(pred_sigma - true_sigma).mean().item()
            sigma_errors.append(sigma_mae)

    val_loss    /= len(val_ds)
    avg_sigma_mae = sum(sigma_errors) / len(sigma_errors)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    elapsed = time.time() - t0
    print(
        f"Epoch [{epoch:3d}/{EPOCHS}]  "
        f"Train: {train_loss:.6f}  "
        f"Val: {val_loss:.6f}  "
        f"Sigma MAE: {avg_sigma_mae:.2f}  "   # ← watch this drop over epochs
        f"({elapsed:.1f}s)"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_loss": val_loss}, CHECKPOINT)
        print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Combined Loss")
plt.title("Blind Denoising — Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nLoss curve saved to {PLOT_PATH}")
print(f"Best val loss: {best_val_loss:.6f}")
print("\nNext step: python evaluate.py")