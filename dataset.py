"""
Each image gets a DIFFERENT random sigma drawn from [NOISE_MIN, NOISE_MAX]
The dataset returns (noisy, clean, sigma) , sigma is needed for the loss
Model never sees the same noise level twice , forced to estimate it
"""

import os
import random
import tarfile
import urllib.request
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ── Config ───────────────────────────────────────────────────────────────────
BSD300_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
RAW_DIR    = Path("data/raw")
CLEAN_DIR  = Path("data/clean")
NOISY_DIR  = Path("data/noisy")
SIGMA_DIR  = Path("data/sigma")       # NEW: stores true sigma per image
SPLITS_DIR = Path("data/splits")

NOISE_MIN  = 10     # minimum sigma (light noise)
NOISE_MAX  = 75     # maximum sigma (heavy noise)
IMG_SIZE   = 128
SEED       = 42


def download_bsd300():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = RAW_DIR / "BSDS300-images.tgz"

    if not tgz_path.exists():
        print("Downloading BSD300 (~130 MB)...")
        urllib.request.urlretrieve(BSD300_URL, tgz_path)
        print("Download complete.")
    else:
        print("Archive already present, skipping download.")

    extract_dir = RAW_DIR / "BSDS300"
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(RAW_DIR)
        print("Extraction complete.")

    img_paths = list(extract_dir.glob("images/**/*.jpg"))
    print(f"Found {len(img_paths)} images in BSD300.")
    return img_paths


def prepare_pairs(img_paths):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_DIR.mkdir(parents=True, exist_ok=True)
    SIGMA_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    random.shuffle(img_paths)

    saved = 0
    for src in img_paths:
        img = cv2.imread(str(src))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        name = f"{saved:05d}.png"

        # Save clean
        cv2.imwrite(str(CLEAN_DIR / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Random sigma for this image — this is the "blind" part
        sigma = random.uniform(NOISE_MIN, NOISE_MAX)

        # Save noisy
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(NOISY_DIR / name), cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))

        # Save true sigma as a single float in a .txt file
        with open(SIGMA_DIR / f"{saved:05d}.txt", "w") as f:
            f.write(str(sigma))

        saved += 1

    print(f"Saved {saved} clean/noisy pairs with random sigma in [{NOISE_MIN}, {NOISE_MAX}].")
    return saved


def make_splits(n_images):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    indices = list(range(n_images))
    random.seed(SEED)
    random.shuffle(indices)

    n_train = int(0.70 * n_images)
    n_val   = int(0.15 * n_images)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(SPLITS_DIR / f"{split}.txt", "w") as f:
            for i in idx:
                f.write(f"{i:05d}.png\n")

    print(f"Split: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test")


class DenoisingDataset(Dataset):
    """
    Returns (noisy_tensor, clean_tensor, sigma_tensor).
    sigma_tensor is a scalar float in the 0-255 scale.
    """

    def __init__(self, split="train"):
        self.to_tensor = transforms.ToTensor()
        split_file     = SPLITS_DIR / f"{split}.txt"
        with open(split_file) as f:
            self.filenames = [line.strip() for line in f]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name  = self.filenames[idx]
        stem  = name.replace(".png", "")

        noisy = cv2.cvtColor(cv2.imread(str(NOISY_DIR / name)), cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(cv2.imread(str(CLEAN_DIR / name)), cv2.COLOR_BGR2RGB)

        with open(SIGMA_DIR / f"{stem}.txt") as f:
            sigma = float(f.read().strip())

        sigma_tensor = torch.tensor([[sigma]], dtype=torch.float32)  # shape (1, 1)

        return self.to_tensor(noisy), self.to_tensor(clean), sigma_tensor


if __name__ == "__main__":
    print("Starting blind dataset preparation...")
    img_paths = download_bsd300()
    n = prepare_pairs(img_paths)
    make_splits(n)
    print("\nDataset ready. Next step: python train.py")