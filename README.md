# NeuroMamba-HG: Anatomically Consistent DFU Segmentation

**Neurosymbolic Vision Mamba with Hypergraph Reasoning for Anatomically Consistent Diabetic Foot Ulcer Segmentation**

Research project exploring four deep learning architectures for diabetic foot ulcer (DFU) image segmentation by combining visual feature extraction, Vision Mamba, hypergraph reasoning, and anatomical constraints.

![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-2457A7) ![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C) ![Models](https://img.shields.io/badge/Model%20Variants-4-6A5ACD) ![Status](https://img.shields.io/badge/Status-Research%20Prototype-555555)

## Overview

Diabetic foot ulcers need regular assessment of their size and boundaries. Manual measurement can be time-consuming and may vary between observers. This project investigates automated wound segmentation that combines image features with long-range context and higher-order relationships between wound regions.

The four model variants share a general pipeline consisting of an encoder, a Vision Mamba bottleneck, hypergraph reasoning, a segmentation decoder, and (in selected variants) symbolic anatomical constraints. The variants explore different encoder backbones, hypergraph construction strategies, and ways of combining visual and anatomical information.

> **Disclaimer:** This is a research prototype, not a clinically validated medical device. Predictions must not be used as the sole basis for diagnosis, treatment, or other clinical decisions.

## Key Features

- **Vision Mamba:** models long-range spatial context using a selective state-space module.
- **Hypergraph Neural Networks:** represent higher-order relationships among multiple image regions.
- **Anatomical reasoning:** selected models incorporate relationships between the wound bed, wound edge, surrounding tissue, and healthy skin.
- **Four model variants:** fixed anatomical hyperedges, dynamically constructed hyperedges, learnable fusion, and hierarchical multi-scale hypergraphs.
- **Evaluation:** Dice, Intersection over Union (IoU), precision, recall, and 95th-percentile Hausdorff distance (HD95).
- **Jupyter notebooks:** model implementation, training, evaluation, and visual analysis.

## Architecture

The overall pipeline is:

1. **Preprocessing:** crop the wound region, resize and normalise images, and augment training samples.
2. **Feature encoder:** extract visual features using the backbone associated with the selected model.
3. **Vision Mamba bottleneck:** capture long-range spatial context.
4. **Hypergraph reasoning:** exchange information between groups of related image nodes.
5. **Symbolic anatomical fusion:** incorporate anatomical constraints in the variants that implement them.
6. **Decoder:** generate a pixel-level wound segmentation mask.
7. **Evaluation:** compare predicted masks against ground-truth masks using overlap and boundary metrics.

## Model Variants

| Model | Notebook | Main design | Input size reported in paper |
|---|---|---|---:|
| Model 1 | `ResNet50_VisionMamba_Hypergraph_Segmentation.ipynb` | ResNet50 encoder, Vision Mamba, fixed anatomical hyperedges, and hyperedge attention | 512 × 512 |
| Model 2 | `ConvNeXt_Mamba_DynamicHypergraph_Segmentation.ipynb` | ConvNeXt Base encoder with dynamically constructed hyperedges based on feature similarity | 512 × 512 |
| Model 3 | `UNet_Mamba_SymbolicHypergraph_Segmentation.ipynb` | Three-stage U-Net with learnable fusion of visual and symbolic information | 384 × 384 |
| Model 4 | `ConvNeXt_VisionMamba_MultigranularityHypergraph_Segmentation.ipynb` | Hierarchical hypergraph reasoning at fine, middle, and coarse scales | 512 × 512 |

## Reported Results

The accompanying paper reports the following test-set metrics. Segmentation metrics were calculated on the 168 wound-positive images among the 229 test images; images without wounds were evaluated separately.

| Model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 ↓ (pixels) |
|---|---:|---:|---:|---:|---:|
| Model 1: ResNet50 + fixed hypergraph | 0.9365 | 0.8838 | **0.9412** | 0.9364 | **21.71** |
| Model 2: ConvNeXt + dynamic hypergraph | **0.9367** | **0.8843** | 0.9341 | 0.9451 | 23.43 |
| Model 3: U-Net + learnable fusion | 0.8759 | 0.7960 | 0.8748 | 0.9001 | 35.21 |
| Model 4: Multi-scale hypergraph | 0.9007 | 0.8305 | 0.8558 | **0.9703** | 28.69 |

**Highlights from the reported results:**

- **Model 2** achieved the highest Dice (0.9367) and IoU (0.8843).
- **Model 1** achieved the highest precision (0.9412) and lowest HD95 (21.71 pixels).
- **Model 4** achieved the highest recall (0.9703).

Dice and IoU measure overlap, while precision and recall measure different aspects of positive pixel predictions. HD95 measures boundary discrepancy, for which lower values are better. These figures are reported by the paper and do not guarantee performance on other datasets or clinical settings.

## Dataset

The study uses 1,370 clinical images from the following datasets:

- Foot Ulcer Segmentation Challenge (FUSC)
- Medetec Foot Ulcer dataset

The paper describes a stratified 70:15:15 split:

| Split | Number of images |
|---|---:|
| Training | 913 |
| Validation | 228 |
| Test | 229 |
| **Total** | **1,370** |

Preprocessing described in the paper includes wound-region cropping, resizing, and ImageNet channel normalisation. Training augmentation includes horizontal and vertical flips, 90-degree rotations, elastic deformation, and colour jitter. Spatial transformations must be applied consistently to each image and its corresponding mask.

### Dataset setup

The notebooks currently reference this Kaggle path:

```text
/kaggle/input/datasets/adnanjan01/wound-seg-3-datasets/data
```

Inspect the data-loading cells and update the configured base path if your dataset is stored elsewhere. The notebooks expect dataset folders containing split directories and corresponding image and label folders. A simplified example is shown below; exact folder names may differ by dataset release.

```text
data/
├── Foot-Ulcer-Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── validation/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── Medetec-Dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── validation/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

The datasets are not included in this repository. Obtain them from their official sources and comply with the relevant licences, access conditions, and research-use requirements.

## Repository Structure

```text
wound-segmentation/
├── README.md
├── ResNet50_VisionMamba_Hypergraph_Segmentation.ipynb
├── ConvNeXt_Mamba_DynamicHypergraph_Segmentation.ipynb
├── UNet_Mamba_SymbolicHypergraph_Segmentation.ipynb
└── ConvNeXt_VisionMamba_MultigranularityHypergraph_Segmentation.ipynb
```

## Requirements

The notebooks use Python and common deep learning and computer-vision packages, including:

- PyTorch and Torchvision
- NumPy
- Pillow
- OpenCV
- Matplotlib
- scikit-learn
- SciPy
- Jupyter Notebook or JupyterLab

A CUDA-capable GPU is recommended for training, especially for the 512 × 512 models. Exact package versions are not pinned in the provided notebooks, so install versions compatible with your Python, PyTorch, Torchvision, CUDA, and GPU environment.

### Example environment setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow opencv-python matplotlib scikit-learn scipy jupyter
```

For GPU training, use the official PyTorch installation instructions to select a build compatible with your system: https://pytorch.org/get-started/locally/

## Getting Started

### Option 1: Run on Kaggle

1. Import or upload the notebooks into Kaggle.
2. Attach the required FUSC and Medetec dataset files.
3. Confirm that the dataset path matches the path configured in the notebook. Update it if necessary.
4. Enable a GPU accelerator if available.
5. Open one of the four model notebooks and run the cells in order.
6. Check the dataset paths and sample image-mask pairs before training.
7. Review validation metrics, test metrics, and predicted-mask visualisations after execution.

### Option 2: Run locally

1. Clone or download the repository.
2. Install the dependencies listed above.
3. Download and arrange the datasets, following their usage requirements.
4. Update the dataset base path in the selected notebook.
5. Start JupyterLab:

   ```bash
   jupyter lab
   ```

6. Open the notebook you want to run and execute the cells in order.

Before training, verify that each image is paired with the correct mask and that the train, validation, and test splits do not overlap. Runtime and memory use depend on the model, hardware, and dataset.

## Training Configuration

The paper describes the following common training settings. Individual notebooks may expose variant-specific settings, so check the configuration cells before running experiments.

| Setting | Configuration described in paper |
|---|---|
| Optimiser | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Maximum training duration | 50 epochs |
| Learning-rate schedule | 5-epoch linear warm-up followed by cosine annealing |
| Gradient clipping | Maximum norm of 1.0 |
| Checkpoint selection | Combined validation Dice and IoU |
| Inference | Six-fold test-time augmentation (TTA) |

The paper also describes early stopping and exponential moving average (EMA) weights. Do not assume that every notebook implements every setting identically; consult its code for the exact behaviour.

## Evaluation

The project reports:

- **Dice coefficient**
- **Intersection over Union (IoU)**
- **Precision**
- **Recall**
- **95th-percentile Hausdorff distance (HD95)**

For a reproducible comparison, use the same dataset version and split, preprocessing, mask conventions, checkpoint selection procedure, and test-time augmentation. Report all relevant metrics and clearly distinguish locally reproduced results from results quoted from the paper.

## Limitations and Reproducibility

- The reported experiments use a combined FUSC and Medetec dataset. Independent external clinical validation is still required.
- Six-fold test-time augmentation increases inference cost and may limit real-time use.
- Model variants have different precision-recall and boundary-performance trade-offs.
- Dataset versions, labels, preprocessing, package versions, hardware, and random seeds may affect results.
- The notebooks are the source of truth for implementation details. This README does not claim that a separate command-line training or inference interface is provided.
- Reported metrics alone do not establish clinical readiness or suitability for patient care.

## Citation

If you use this work in academic research, cite the associated paper. Update this entry with publication or preprint details when they become available.

```text
Ahmad, S., Iqbal, H., Farooq, J. A., and Panda, G.
"Neurosymbolic Vision Mamba with Hypergraph Reasoning for Anatomically
Consistent Diabetic Foot Ulcer Segmentation."
C.V. Raman Global University.
```

## Authors

- Shayam Ahmad
- Haroon Iqbal
- Jan Adnan Farooq
- Ganapati Panda

**Affiliation:** C.V. Raman Global University

## Responsible Use

This project is intended for academic research and technical experimentation. It has not been established as a clinically validated diagnostic system. Do not use its predictions as a substitute for professional medical assessment. Any clinical application would require appropriate external validation, risk assessment, regulatory review, and oversight by qualified healthcare professionals.

## Acknowledgements

The work uses the Foot Ulcer Segmentation Challenge (FUSC) and Medetec foot-ulcer image datasets. Please acknowledge the dataset creators and comply with their respective terms of use when using the data.
