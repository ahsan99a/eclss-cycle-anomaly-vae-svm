# eclss-cycle-anomaly-vae-svm
Cycle-aware machine learning model (VAE + SVM) for anomaly detection and fault classification in synthetic ECLSS CO₂-removal cycles.  
**Synthetic Sensor Dataset • Anomaly Detection • Fault Classification**
## 🔭 Overview

This repository contains an end-to-end machine learning pipeline for **fault detection and diagnosis** in NASA-inspired **Environmental Control and Life Support System (ECLSS)** data.  
It includes:

- A fully simulated **multi-sensor dataset** (O₂, CO₂, Pressure)
- A **β-Variational Autoencoder (VAE)** for anomaly detection  
- An **SVM classifier** trained on latent features for fault identification  
- Complete **EDA**, validation routines, and publication-quality plots

The project demonstrates how deep learning can support **reliability & health monitoring** for deep-space habitat systems.

## 🚀 Key Features

### ✔ Synthetic Dataset  
Includes **1000-step cycles** with 3 sensors:
- O₂ (%)
- CO₂ (%)
- Pressure (psi)

Six system states simulated:
1. Nominal  
2. CO₂ Leak  
3. Valve Stiction  
4. Vacuum Anomaly  
5. CDRA Degradation  
6. OGA Degradation  

Plus **sensor fault models**:
- Bias drift  
- High noise  
- Partial freeze  
- Spike outliers  

All samples tagged with **safety flags** + metadata.

### ✔ β-Variational Autoencoder (VAE)

Architecture:
- Encoder: `3000 → 1024 → 512 → 256 → (μ, logσ²)`
- Decoder mirrors encoder  
- Latent space: **32-dimensional**  
- Loss: **MSE + β KL-divergence (β = 0.3–0.5)**

Outputs:
- Reconstruction errors  
- Latent vectors (μ)  
- ROC curve, thresholding metrics  
- Saved PyTorch model (`vae_dense_eclss.pth`)

**VAE Performance**
| Metric | Value |
|--------|--------|
| Train accuracy | **98.81%** |
| AUC | **≈ 0.8491** |

### ✔ SVM Fault-Type Classifier (Using VAE Latent Space)

- Kernel: **RBF**
- Tuned via grid search  
- Inputs: latent vectors from VAE  

**SVM Performance**
| Metric | Value |
|--------|--------|
| Train accuracy | **99.5%** |
| Test accuracy | **95.6%** |
| Classes | 5 fault modes |



