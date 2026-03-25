# Blind Image Denoising using U-Net

## Overview

This project implements a deep learning-based approach for **blind image denoising** using a modified U-Net architecture. The model is designed to simultaneously:

1. Remove noise from corrupted images
2. Estimate the noise level (σ) directly from the input image

Unlike traditional denoising methods, the model does not require prior knowledge of the noise level, making it suitable for real-world scenarios.

---

## Key Features

* Blind denoising (no prior noise information required)
* Joint learning of image restoration and noise estimation
* U-Net architecture with skip connections for precise reconstruction
* Trained on natural images (BSD300 dataset)
* Evaluation using PSNR, SSIM, and MAE

---

## Methodology

### Dataset

* Source: BSD300 dataset
* Images are corrupted using Gaussian noise with σ randomly sampled from [10, 75]
* Dataset split:

  * Training: 70%
  * Validation: 15%
  * Test: 15%

### Model Architecture

The model is based on a U-Net encoder–decoder structure:

* **Encoder**: Extracts hierarchical features through downsampling
* **Bottleneck**: Captures global image representation
* **Decoder**: Reconstructs the clean image
* **Skip Connections**: Preserve spatial details by linking encoder and decoder layers

A separate **noise estimation head** is attached at the bottleneck to predict σ.

---

## Loss Function

The model is trained using a combined loss:

* Image reconstruction loss: Mean Squared Error (MSE)
* Noise estimation loss: Weighted MSE

Total Loss:

```
Loss = MSE_denoising + 0.01 × MSE_sigma
```

---

## Results

The model demonstrates effective denoising across varying noise levels.

Typical performance:

* Significant improvement in PSNR over noisy inputs
* SSIM values indicating good structural preservation
* Accurate estimation of noise level (σ)

Note: Outputs may appear slightly smooth due to the use of MSE loss.

---

## Project Structure

```
unet_denoising/
│
├── dataset.py          # Data loading and noise generation
├── model.py            # U-Net architecture with noise estimation head
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation metrics (PSNR, SSIM, MAE)
├── predict.py          # Inference and visualization
├── requirements.txt    # Dependencies
├── README.md
│
├── data/               # Dataset (not included)
├── checkpoints/        # Saved models
├── outputs/            # Predictions and visual results
```

---

## Installation

Clone the repository:

```
git clone https://github.com/anshuliii/BlindDenoising_UNet.git
cd BlindDenoising_UNet
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### Train the model

```
python train.py
```

### Evaluate the model

```
python evaluate.py
```

### Run inference

```
python predict.py
```

---

## Future Work

* Incorporate perceptual loss to reduce blurriness
* Explore GAN-based approaches for sharper outputs
* Extend to real-world noise distributions
* Optimize model for faster inference

---

## License

This project is for academic and research purposes.
