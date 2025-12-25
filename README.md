# Hybrid Statistical-Autoencoder Model for Adaptive IoT Node Authentication in 5G Networks

This project addresses the problem of IoT device authentication by leveraging radio frequency (RF) fingerprints that capture unique hardware-induced signal variations instead of relying on cryptographic credentials.

## Project Structure

```
├── data/                      # RadioML 2018.01A datasets
│   ├── raw/                   # Unprocessed IQ data
│   └── processed/             # Legitimate vs. Attack samples
├── src/                       # Source code modules
│   ├── models/                
│   │   ├── autoencoder.py     # Encoder/Decoder architectures
│   │   └── classifier.py      # OC-SVM, Gaussian, Isolation Forest
│   ├── utils/                 
│   │   ├── preprocessing.py   # Normalization and I/Q concatenation
│   │   └── metrics.py          # AUC, F1, FAR, FRR calculation logic
│   ├── train_ae.py            # Phase 1: Training the Autoencoder
│   └── train_classifier.py    # Phase 2: Training the One-Class models
├── notebooks/                 # Google Colab notebooks
│   ├── 01_Feature_Learning.ipynb
│   └── 02_Anomaly_Detection.ipynb
├── saved_models/              # Serialized model weights
│   ├── ae_weights.pth
│   └── classifier_model.joblib
└── README.md                  # Project overview and instructions
```

## Overview

This project implements a **Hybrid Statistical-Autoencoder Model** for Open Set Recognition (OSR) in IoT device authentication. The system consists of two phases:

### Phase 1: Feature Learning (Autoencoder)
- Deep Convolutional Autoencoder with Squeeze-and-Excitation (SE) blocks
- Joint Contrastive Learning (SimCLR) for robust feature extraction
- Extracts 64-dimensional latent representations from RF signals

### Phase 2: Anomaly Detection (One-Class Classifiers)
- **OC-SVM**: One-Class Support Vector Machine (baseline)
- **ECOD**: Empirical Cumulative Distribution-based Outlier Detection (SOTA 2024)
- **EVT**: Extreme Value Theory with Peaks-Over-Threshold for robust thresholding

## Features

- **RF Fingerprinting**: Captures unique hardware impairments from I/Q signals
- **Open Set Recognition**: Detects unknown/unauthorized devices
- **State-of-the-Art Methods**: Incorporates latest 2024-2025 research
- **GPU Support**: Optimized for CUDA acceleration
- **Colab Ready**: Includes notebooks for Google Colab training

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
pyod>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
h5py>=3.0.0
joblib>=1.0.0
tqdm>=4.60.0
```

## Quick Start

### Local Training

1. **Preprocess Data**:
```bash
python src/utils/preprocessing.py
```

2. **Train Autoencoder (Phase 1)**:
```bash
python src/train_ae.py
```

3. **Train Classifiers (Phase 2)**:
```bash
python src/train_classifier.py
```

### Google Colab Training

See `notebooks/` directory for Colab-ready notebooks:
- `01_Feature_Learning.ipynb` - Phase 1 training with GPU
- `02_Anomaly_Detection.ipynb` - Phase 2 training

## Dataset

This project uses **RadioML 2018.01A** dataset:
- 24 modulation classes
- SNR range: -20 dB to +18 dB
- I/Q samples: 1024 samples per signal
- Classes 0-4 (OOK, 4ASK, 8ASK, BPSK, QPSK) treated as legitimate devices

## Model Architecture

### Autoencoder
- **Input**: (Batch, 2, 128) I/Q samples
- **Encoder**: 3x Conv1D + SE Attention + MaxPool
- **Latent**: 64-dim L2-normalized vector
- **Decoder**: 3x ConvTranspose1D

### Classifiers
- **OC-SVM**: RBF kernel, nu=0.1
- **ECOD**: Parameter-free, contamination=0.01
- **EVT**: POT method with FRR=0.01

## Results

The model achieves:
- High True Positive Rate (TPR) for legitimate devices
- Low False Positive Rate (FPR) for unauthorized devices
- Robust performance across different SNR conditions

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid-statistical-autoencoder-iot-auth,
  title={Hybrid Statistical-Autoencoder Model for Adaptive IoT Node Authentication in 5G Networks},
  author={Thu Hoang Anh, Hung Nguyen Le},
  year={2025},
  url={https://github.com/lavonaschaffer/Hybrid-Statistical-Autoencoder-Model-for-Adaptive-IoT-Node-Authentication-in-5GNetworks}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- RadioML 2018.01A dataset
- PyOD library for anomaly detection methods
- PyTorch community

