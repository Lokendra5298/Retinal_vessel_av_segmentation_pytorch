# Retinal Blood Vessel Segmentation using UNet, TransUnet and GAN

This repository contains multiple deep learning approaches for **retinal blood vessel segmentation** using fundus images.

The project compares:

* 🔹 U-Net
* 🔹 U-Net++
* 🔹 ResNetV2-based segmentation (TransUNet-style CNN)
* 🔹 GAN-based segmentation (Pix2Pix-style)

The models are trained and evaluated on multiple public retinal datasets to improve generalization.

---

# 📂 Datasets

The following public datasets are used:

* **DRIVE**
* **CHASE DB1**
* **HRF**
* **FIVES**
* RETA dataset

### Dataset Usage

| Dataset   | Usage                              |
| --------- | ---------------------------------- |
| RETA      | Training                           |
| CHASE DB1 | Training                           |
| HRF       | Training                           |
| FIVES     | Training                           |
| DRIVE     | Testing (cross-dataset evaluation) |

>  Datasets must be downloaded manually from their official websites due to licensing restrictions.

---

# 🏗 Implemented Models

---

## 1️⃣ U-Net & U-Net++

Implemented using **segmentation_models_pytorch**

* Encoder: **EfficientNet-B4** (ImageNet pretrained)
* Patch-based training
* Loss: BCE + Dice
* Overlapping patch reconstruction

---

## 2️⃣ ResNetV2 Segmentation Model (TransUNet-style CNN)

* Custom implementation of **ResNet** (Pre-activation)
* GroupNorm instead of BatchNorm
* Progressive bilinear decoder
* Loss: CrossEntropy + Dice
* Metric: Dice + HD95

> Note: This model does NOT include transformer layers.

---

## 3️⃣ GAN-based Segmentation (RVGAN)

Pix2Pix-style conditional GAN:

* Generator: U-Net
* Discriminator: PatchGAN
* Loss:

  * Adversarial loss
  * L1 loss
  * Dice loss
* Mixed precision training

---

# 📊 Evaluation Metrics

All models are evaluated using:

* Dice Score
* IoU (mIoU)
* Accuracy
* Sensitivity (Recall)
* Specificity
* AUC
* HD95 (Hausdorff Distance)

These metrics are commonly used in medical image segmentation.

---

# ⚙️ Project Structure

```
.
├── Models_large_data.ipynb      # U-Net & U-Net++
├── TransUNet.ipynb              # ResNetV2 segmentation
├── GAN_Model.ipynb              # RVGAN implementation
├── datasets/
│   ├── DRIVE/
│   ├── CHASEDB1/
│   ├── HRF/
│   ├── FIVES/
│   └── RETA/
└── checkpoints/
```

---

# 🔄 Data Processing Pipeline

1. Load image-mask pairs
2. Apply joint augmentations
3. Resize to 512×512
4. Extract overlapping patches (128×128, stride=64)
5. Remove empty patches (optional)
6. Normalize images
7. Convert masks to binary

---

# 🧠 Training Details

### Optimizer

* AdamW

### Learning Rate

* 1e-4 or 1e-3 depending on model

### Scheduler

* ReduceLROnPlateau
* CosineAnnealingLR (ResNet model)

### Mixed Precision

* Used in GAN and ResNet training

### Checkpointing

* Best model saved based on validation Dice or mIoU

---

# 🚀 How to Run

## 1️⃣ Install Dependencies

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install scikit-learn
pip install scipy
```

---

## 2️⃣ Prepare Dataset

Place datasets in:

```
datasets/
    DRIVE/
    CHASEDB1/
    HRF/
    FIVES/
    RETA/
```

Ensure images and masks are correctly paired.

# 📈 Expected Outputs

* Training loss curves
* Validation Dice score
* Reconstructed full-image predictions
* Quantitative evaluation metrics
* Visual comparison plots
* Or restructure this into a clean GitHub-ready research repository layout**
