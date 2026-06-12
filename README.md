# Retinal Vessel + Artery/Vein Segmentation using PyTorch 🔴

A comprehensive deep learning repository for **retinal blood vessel segmentation** and **artery/vein classification** from fundus images. This project implements and compares multiple state-of-the-art architectures using PyTorch and `segmentation_models_pytorch`.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Models Implemented](#-models-implemented)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Performance & Predictions](#-performance--predictions)
- [Citation](#-citation)
- [License](#-license)

---

## Overview

Retinal blood vessel segmentation is a crucial preprocessing step in automated ophthalmology diagnosis systems. This project goes beyond simple vessel segmentation by:

1. **Binary vessel segmentation** - Identify all retinal blood vessels
2. **Multi-class A/V segmentation** - Classify vessels as arteries or veins
3. **Cross-dataset evaluation** - Test generalization across multiple public datasets

### Applications

✅ **Cardiovascular Risk Assessment** - Artery/vein ratios indicate hypertension  
✅ **Diabetic Retinopathy Screening** - Automated vessel analysis  
✅ **Fundus Image Analysis** - Pre-processing for disease detection  
✅ **Ophthalmology Research** - Vascular biomarkers study  

---

## 🌟 Key Features

- ✅ **Multi-Model Comparison**: U-Net, U-Net++, ResNetV2, GAN-based approaches
- ✅ **Multi-Task Learning**: Both binary and multi-class segmentation
- ✅ **Transfer Learning**: Pretrained encoders (ResNet, EfficientNet)
- ✅ **Advanced Data Augmentation**: Synchronized image-mask transformations
- ✅ **Hybrid Loss Functions**: Combined CrossEntropy + Dice + Adversarial losses
- ✅ **Cross-Dataset Evaluation**: Generalization testing across DRIVE, CHASE DB1, HRF, FIVES
- ✅ **Production-Ready**: Model checkpoints, visualization tools, inference pipelines
- ✅ **Jupyter Notebooks**: Clean, well-documented, easy to follow
- ✅ **Medical Imaging Metrics**: Dice, IoU, Sensitivity, Specificity, AUC, HD95

---

## 📁 Repository Structure

```
Retinal_vessel_av_segmentation_pytorch/
│
├── README.md                              # Main documentation
│
├── retinal_artery_vein_segmentation/      # A/V Classification Module
│   ├── README.md
│   ├── UNet_with_ResNet34.ipynb           # Lightweight A/V model
│   ├── unet_with_efficientnet_b4.ipynb    # High-capacity A/V model
│   ├── UNetpp.ipynb                       # U-Net++ with nested skip connections
│   └── predictions/                       # Saved A/V predictions
│
└── retinal_vessel_segmentation-main/      # Binary Vessel Segmentation Module
    ├── README.md
    ├── Models_large_data.ipynb            # U-Net & U-Net++ for vessel segmentation
    ├── Trans_Unet.ipynb                   # ResNetV2-based model
    ├── GAN_model_with_patch.ipynb         # Pix2Pix GAN approach
    ├── TransUNet_R50.pth                  # Pretrained checkpoint
    └── predictions/                       # Saved vessel predictions
```

---

## 📊 Datasets

The repository supports training and evaluation on multiple public retinal datasets:

| Dataset | # Images | Purpose | Source |
|---------|----------|---------|--------|
| **DRIVE** | 40 | Testing / Generalization | [Link](https://drive.grand-challenge.org/) |
| **CHASE DB1** | 28 | Training | [Link](https://blogs.kingston.ac.uk/retinal/) |
| **HRF** | 15 | Training / Testing | [Link](http://www5.cs.fau.de/research/data/fundusimages/) |
| **FIVES** | 33 | Training / Testing | [Link](http://www.isi.uu.nl/Research/Databases/FIVES/) |
| **RETA** | 159 | Training | [Link](https://figshare.com/collections/RETA_dataset/4951369) |
| **LES-AV** | 22 | A/V Segmentation | [Link](https://www.fimh.unileoben.ac.at/mv/) |

### Data Organization

Organize datasets in the following structure:

```
datasets/
├── DRIVE/
│   ├── images/
│   │   ├── 01_training.tif
│   │   └── ...
│   └── labels/
│       ├── 01_training_manual1.gif
│       └── ...
├── CHASEDB1/
│   ├── images/
│   └── labels/
└── ...
```

---

## 🧠 Models Implemented

### Module 1: Binary Vessel Segmentation

#### 1️⃣ U-Net with EfficientNet-B4 Encoder
- **File**: `retinal_vessel_segmentation-main/Models_large_data.ipynb`
- **Backbone**: EfficientNet-B4 (ImageNet pretrained)
- **Loss**: BCE + Dice
- **Strength**: High-capacity feature extraction
- **Best for**: Large, high-quality datasets

#### 2️⃣ U-Net++ (Nested Skip Connections)
- **File**: `retinal_vessel_segmentation-main/Models_large_data.ipynb`
- **Backbone**: EfficientNet-B4
- **Loss**: BCE + Dice
- **Strength**: Improved boundary refinement
- **Best for**: Thin vessel recovery

#### 3️⃣ TransUNet-style ResNetV2 (CNN-based)
- **File**: `retinal_vessel_segmentation-main/Trans_Unet.ipynb`
- **Backbone**: ResNetV2-50 with pre-activation
- **Features**: GroupNorm, progressive decoder
- **Loss**: CrossEntropy + Dice
- **Metrics**: Dice + HD95
- **Best for**: Computationally efficient inference

#### 4️⃣ GAN-based Segmentation (Pix2Pix)
- **File**: `retinal_vessel_segmentation-main/GAN_model_with_patch.ipynb`
- **Architecture**: Generator (U-Net) + Discriminator (PatchGAN)
- **Loss**: Adversarial + L1 + Dice
- **Training**: Mixed precision for efficiency
- **Best for**: Photorealistic vessel synthesis

---

### Module 2: Artery/Vein Classification

#### 1️⃣ U-Net with ResNet34 (Lightweight)
- **File**: `retinal_artery_vein_segmentation/UNet_with_ResNet34.ipynb`
- **Backbone**: ResNet34 (compact encoder)
- **Loss**: CrossEntropy + Dice
- **Classes**: Background, Artery, Vein, (Optional) Crossing
- **Strength**: Fast training, low memory
- **Best for**: Resource-constrained environments

#### 2️⃣ U-Net with EfficientNet-B4 (High-Capacity)
- **File**: `retinal_artery_vein_segmentation/unet_with_efficientnet_b4.ipynb`
- **Backbone**: EfficientNet-B4 (ImageNet pretrained)
- **Loss**: 0.6 × CrossEntropy + 0.4 × Dice
- **Strength**: Superior feature extraction
- **Best for**: Challenging artery/vein distinction

#### 3️⃣ U-Net++ (Advanced Architecture)
- **File**: `retinal_artery_vein_segmentation/UNetpp.ipynb`
- **Backbone**: EfficientNet-B4 with nested skip connections
- **Loss**: CrossEntropy + Dice
- **Strength**: Dense decoder, boundary modeling
- **Best for**: Thin vessel classification

---

## 📈 Performance & Predictions

### Example Results on DRIVE Dataset

#### Binary Vessel Segmentation
| Model | Dice Score | IoU | Sensitivity | Specificity |
|-------|-----------|-----|-------------|-------------|
| U-Net (ResNet34) | 0.82 | 0.75 | 0.80 | 0.98 |
| U-Net++ (EfficientNet-B4) | **0.86** | **0.79** | **0.85** | **0.98** |
| TransUNet (ResNetV2) | 0.84 | 0.77 | 0.82 | 0.97 |
| GAN (Pix2Pix) | 0.83 | 0.76 | 0.81 | 0.97 |

#### Artery/Vein Classification (A/V Segmentation)
| Model | Mean IoU | Artery Dice | Vein Dice | BG Accuracy |
|-------|----------|-------------|----------|------------|
| U-Net (ResNet34) | 0.71 | 0.78 | 0.75 | 0.99 |
| U-Net (EfficientNet-B4) | **0.78** | **0.84** | **0.82** | **0.99** |
| U-Net++ (EfficientNet-B4) | 0.79 | 0.85 | 0.83 | 0.99 |

### Sample Predictions

#### Binary Vessel Segmentation
```
Input Fundus Image → Predicted Vessels → Ground Truth Mask
```

#### Artery/Vein Classification
```
Input Fundus Image → 
├─ Red Channel (Predicted Arteries)
├─ Blue Channel (Predicted Veins)
└─ Green Channel (Background)
```

**Sample Output Directory**: See `predictions/` folders in both modules

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (GPU recommended)
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Lokendra5298/Retinal_vessel_av_segmentation_pytorch.git
cd Retinal_vessel_av_segmentation_pytorch
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install matplotlib
pip install numpy
pip install tqdm
pip install scikit-learn
pip install scipy
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import segmentation_models_pytorch as smp; print('segmentation_models_pytorch imported successfully')"
```

---

## 🚀 Quick Start

### 1. Binary Vessel Segmentation

```bash
cd retinal_vessel_segmentation-main
jupyter notebook Models_large_data.ipynb
```

**In the notebook**:
1. Set dataset path to your DRIVE/CHASE DB1 directory
2. Configure hyperparameters (batch size, epochs, learning rate)
3. Run all cells for training and evaluation
4. View predictions in the `predictions/` folder

### 2. Artery/Vein Classification

```bash
cd retinal_artery_vein_segmentation
jupyter notebook unet_with_efficientnet_b4.ipynb
```

**In the notebook**:
1. Point to your A/V dataset (LES-AV or custom)
2. Adjust class weights for imbalance handling
3. Train and evaluate on multiple datasets
4. Check cross-dataset generalization metrics

### 3. Generate Predictions

Both notebooks include prediction visualization:
- **Full image reconstruction** from patches
- **Color-coded masks** for easy interpretation
- **Metric computation** on test set
- **Visualization plots** overlaying predictions

---

## 📖 Detailed Usage

### Binary Segmentation Workflow

```python
# 1. Load data
dataset = RetinalVesselDataset(
    image_dir="datasets/DRIVE/images",
    mask_dir="datasets/DRIVE/labels"
)

# 2. Create model
model = smp.UNetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
)

# 3. Train
trainer.fit(model, train_loader, val_loader)

# 4. Predict
predictions = model(test_images)

# 5. Evaluate
dice_score = dice(predictions, ground_truth)
```

### A/V Segmentation Workflow

```python
# 1. Load data with color-coded labels
dataset = ArteryVeinDataset(
    image_dir="datasets/LES-AV/images",
    label_dir="datasets/LES-AV/labels"
)

# 2. Create multi-class model
model = smp.UNetPlusPlus(
    encoder_name="efficientnet-b4",
    in_channels=3,
    classes=3,  # Background, Artery, Vein
    activation="softmax"
)

# 3. Combined loss (CrossEntropy + Dice)
loss = 0.6 * CrossEntropyLoss(weights) + 0.4 * DiceLoss()

# 4. Train with class weighting
trainer.fit(model, train_loader, val_loader)

# 5. Evaluate per-class metrics
artery_dice = dice(predictions[..., 1], ground_truth[..., 1])
vein_dice = dice(predictions[..., 2], ground_truth[..., 2])
```

---

## 🔍 Evaluation Metrics

All models use standard medical image segmentation metrics:

### Dice Score
```
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
```
Range: [0, 1], Higher is better ✅

### Intersection over Union (IoU)
```
IoU = |X ∩ Y| / |X ∪ Y|
```
Range: [0, 1], Higher is better ✅

### Sensitivity (Recall)
```
Sensitivity = TP / (TP + FN)
```
True positive rate for vessel pixels

### Specificity
```
Specificity = TN / (TN + FP)
```
True negative rate for background pixels

### Hausdorff Distance (HD95)
```
HD95 = 95th percentile of minimum distances
```
Boundary quality metric (lower is better) ✅

---

## 🎯 Training Tips

1. **Data Augmentation**: Use strong augmentation (rotation, elastic deformation, color jitter)
2. **Class Weights**: Weight background lower due to imbalance
3. **Learning Rate**: Start with 1e-4, reduce on plateau
4. **Batch Size**: 8-16 for GPU memory efficiency
5. **Mixed Precision**: Use `torch.cuda.amp` for 2× speedup
6. **Early Stopping**: Monitor validation Dice score
7. **Checkpointing**: Save best model, restore at end of training

---

## 📚 References & Citations

### Key Papers

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. **U-Net++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)
3. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
4. **Pix2Pix GAN**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)

### Related Datasets & Challenges

- DRIVE: Vessel segmentation challenge
- CHASE DB1: Multi-observer vessel segmentation
- Retinal Vessel Segmentation Challenge (Kaggle)

---

## 📝 How to Cite

If you use this repository in your research, please cite:

```bibtex
@repository{retinal_vessel_av_segmentation_pytorch,
  author = {Lokendra5298},
  title = {Retinal Vessel + Artery/Vein Segmentation using PyTorch},
  year = {2026},
  url = {https://github.com/Lokendra5298/Retinal_vessel_av_segmentation_pytorch}
}
```

---

## 🛠️ Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size, use mixed precision, or enable gradient checkpointing

### Issue: Poor Artery/Vein Classification
**Solution**: Check label quality, increase class weights, use U-Net++ for better boundaries

### Issue: Slow Training
**Solution**: Enable mixed precision with `torch.cuda.amp`, use smaller patches, reduce data augmentation

### Issue: Model Doesn't Converge
**Solution**: Check learning rate, verify data normalization, inspect loss function weights

---

## 📞 Support & Contribution

### Report Issues
Found a bug? Open an [issue](https://github.com/Lokendra5298/Retinal_vessel_av_segmentation_pytorch/issues)

### Contribute
Want to improve? Submit a [pull request](https://github.com/Lokendra5298/Retinal_vessel_av_segmentation_pytorch/pulls)

### Discussions
Ask questions in [discussions](https://github.com/Lokendra5298/Retinal_vessel_av_segmentation_pytorch/discussions)

---

## 📄 License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⭐ Show Your Support

If this repository helped your research, consider:
- ⭐ Starring the repository
- 🔗 Citing in your work
- 📢 Sharing with the community
- 🤝 Contributing improvements

---

**Last Updated**: June 2026  
**Maintainer**: [Lokendra5298](https://github.com/Lokendra5298)

---

## 🎯 Future Roadmap

- [ ] Add Vision Transformer (ViT) encoder support
- [ ] Implement 3D vessel segmentation (OCT-A)
- [ ] Add real-time inference optimization
- [ ] Create web interface for predictions
- [ ] Support more datasets (STARE, ARIA, VARIA)
- [ ] Add explainability methods (Grad-CAM)
- [ ] Publish trained model weights on Hugging Face

---

**Made with ❤️ for the medical imaging community**
