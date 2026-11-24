# CT-to-MRI Translation with Voxel-wise Reliability Estimation

## Project Overview

This project develops a deep learning-based system for translating brain CT scans into MRI-like images, accompanied by a voxel-wise reliability map.

The system combines image-to-image translation, Test-Time Augmentation (TTA)-based uncertainty estimation, and example-based reference retrieval to provide additional information about the generated MRI.

The objective is to generate MRI-like images from CT scans and identify regions where the generated output may be less reliable.

## Key Features

- **CT-to-MRI Translation:** Generates MRI-like brain images from CT volumes using deep learning.
- **Residual U-Net:** Baseline image-to-image translation model.
- **Pix2Pix:** GAN-based image translation using a Residual U-Net generator and PatchGAN discriminator.
- **Four-view Test-Time Augmentation:** Uses original, horizontal flip, vertical flip, and combined horizontal-vertical flip predictions.
- **Voxel-wise Uncertainty Estimation:** Estimates prediction disagreement across TTA views.
- **Reliability Map Generation:** Produces a spatial uncertainty map alongside the generated MRI.
- **EvidenceMap:** Retrieves similar CT patches and their paired MRI references from the training dataset.
- **Evaluation:** Uses MAE, SSIM, PSNR, Spearman correlation, and automated anatomical Dice evaluation.

## System Architecture

### 1. Data Preparation

The project uses paired CT and MRI volumes with corresponding masks.

Preprocessing includes:

- CT intensity clipping and normalization.
- MRI percentile-based normalization.
- Volume and affine alignment checks.
- Patient-level training and validation splitting.
- Mask-based evaluation.

### 2. CT-to-MRI Translation

Two image translation approaches were implemented.

**Model 1: Residual U-Net**

- Encoder-decoder architecture.
- Residual learning blocks.
- Three-channel CT input using neighboring slices.
- L1 reconstruction loss.
- AdamW optimizer.

**Model 2: Pix2Pix**

- Residual U-Net generator.
- PatchGAN discriminator.
- Adversarial loss combined with L1 reconstruction loss.
- Separate generator and discriminator optimization.

### 3. Reliability Estimation using Four-view TTA

Four predictions are generated using:

1. Original CT input.
2. Horizontal flip.
3. Vertical flip.
4. Combined horizontal and vertical flip.

Predictions are transformed back to the original orientation.

The mean prediction is used as the final TTA output, while voxel-wise standard deviation across the predictions is used as an uncertainty estimate.

The uncertainty map is normalized to the range 0–1 for visualization.

Higher values indicate greater disagreement among TTA predictions.

**Note:** TTA disagreement is an uncertainty proxy, not a calibrated probability of correctness.

### 4. EvidenceMap Reference Retrieval

EvidenceMap provides example-based CT-to-MRI reference retrieval.

- 115 training patients are used to construct the patch index.
- Paired CT and MRI patches of size 32 × 32 are extracted.
- Total indexed patch pairs: 490,404.
- Normalized Cross-Correlation (NCC) is used for similarity matching.
- Top three similar CT patches are retrieved.
- Corresponding paired MRI references are displayed.
- The query patient is excluded from retrieval.

EvidenceMap provides reference examples and does not generate the synthetic MRI.

## Dataset

Dataset: CIT Hackathon PS-07 CT-to-MRI Dataset

Source: https://huggingface.co/datasets/Tarakeshwaran/CIT_Hackathon_PS07_ct2mri

Dataset characteristics:

- 120 paired CT/MRI volumes.
- 115 training patients.
- 5 validation patients.
- Two training centres.
- CT, MRI and mask volumes.
- Volume dimensions: 192 × 224 × 192.

The five validation patients are:

- P2658
- P6123
- P7363
- P7480
- P8947

The validation patients belong to Centre B. Therefore, this validation setup should not be described as an independent unseen-hospital test.

**Dataset license:** CC BY-NC 4.0. Follow the dataset license and attribution requirements. This project is intended for non-commercial research and educational use.

## Evaluation Results

### A. Image Translation Evaluation

Results averaged across the five validation patients.

| Metric | Residual U-Net | Pix2Pix |
|---|---:|---:|
| Full-volume MAE | 0.06649 | 0.06407 |
| Masked MAE | 0.11183 | 0.11621 |
| Mean slice SSIM | 0.56352 | 0.62953 |
| PSNR (dB) | 17.9305 | 17.5743 |

The Pix2Pix model achieved higher mean SSIM, while the Residual U-Net achieved lower masked MAE and higher PSNR.

### B. TTA Reliability Evaluation

| Patient | Masked MAE | Spearman Correlation |
|---|---:|---:|
| P2658 | 0.10908 | 0.57988 |
| P6123 | 0.11643 | 0.56738 |
| P7363 | 0.11914 | 0.52960 |
| P7480 | 0.11521 | 0.56416 |
| P8947 | 0.10744 | 0.59302 |
| **Average** | **0.11346** | **0.56681** |

The positive Spearman correlations indicate that higher TTA disagreement tends to be associated with higher reconstruction error across the evaluated voxels.

### C. Anatomical Evaluation

SynthSeg Robust 2.0 was used to obtain automated segmentations for the ground-truth and generated MRI volumes.

Mean patient-level Dice scores:

| Model | Mean Dice |
|---|---:|
| Residual U-Net | 0.2451 |
| Pix2Pix | 0.1117 |

These results represent automated anatomical agreement and should not be interpreted as clinical validation.

## Technology Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- NiBabel
- SciPy
- Scikit-image
- Scikit-learn
- Jupyter Notebook
- SynthSeg Robust 2.0
- Visual Studio Code

## Project Structure

```text
PS07_CT2MRI/
│
├── dataset/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── evidence_map/
│   ├── reliability_test/
│   ├── validation/
│   └── visualizations/
│
├── results/
│   ├── reliability/
│   └── synthseg/
│
├── src/
│   ├── evidence_map/
│   │   ├── build_patch_index.ipynb
│   │   ├── retrieve_examples.py
│   │   └── visualize_evidence.ipynb
│   │
│   ├── infer_pix2pix_best.py
│   ├── patchgan_discriminator.py
│   └── resunet.py
│
├── train_patients.csv
├── val_patients.csv
├── .gitignore
├── requirements.txt
└── README.md
```

The actual repository may contain additional notebooks and result files.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PS07_CT2MRI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch

Install a PyTorch version compatible with your hardware and CUDA configuration using the official PyTorch installation instructions.

### 5. Prepare the dataset

Download the dataset from the Hugging Face dataset repository and place it in the expected `dataset/volumes/` structure.

### 6. Run the notebooks

Open the project in Visual Studio Code and execute the notebooks in the appropriate order.

## Limitations

- The validation set contains only five patients.
- Validation patients belong to Centre B; independent unseen-hospital generalization has not been established.
- TTA disagreement is not a calibrated confidence estimate.
- EvidenceMap retrieves reference examples rather than generating patient-specific MRI anatomy.
- Automated SynthSeg Dice results are limited and do not establish clinical accuracy.
- The system is a research prototype and is not intended for clinical diagnosis.

## Future Improvements

- Evaluate on a genuinely independent hospital dataset.
- Improve anatomical structure preservation.
- Investigate calibrated uncertainty estimation.
- Expand reference retrieval across multiple anatomical regions.
- Evaluate additional image quality and anatomical metrics.
- Optimize inference for deployment.

## Disclaimer

This project is developed for academic research and educational purposes as part of the CIT Hackathon PS-07 challenge.

Generated MRI-like images and reliability maps must not be used as substitutes for actual diagnostic MRI scans or clinical decisions.
