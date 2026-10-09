# NeuroMamba-HG: Anatomically Consistent Diabetic Foot Ulcer Segmentation

```{=html}
<p align="center">
```
`<strong>`{=html}Neurosymbolic Vision Mamba with Hypergraph Reasoning
for Anatomically Consistent Diabetic Foot Ulcer
Segmentation`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
Research implementations of four neurosymbolic deep learning
architectures for automated diabetic foot ulcer (DFU) image
segmentation.
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-2457A7" alt="Task: Medical Image Segmentation">`{=html}
`<img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" alt="Framework: PyTorch">`{=html}
`<img src="https://img.shields.io/badge/Models-4-6A5ACD" alt="Four model variants">`{=html}
`<img src="https://img.shields.io/badge/Status-Research%20Prototype-555555" alt="Research prototype">`{=html}
```{=html}
</p>
```
## Overview

Diabetic foot ulcers require regular assessment of wound size and
boundaries. Manual assessment can be time-consuming and may vary between
observers. This project investigates automated wound segmentation by
combining visual feature learning with long-range context, hypergraph
reasoning, and anatomical knowledge.

The repository contains four notebook-based model implementations. Each
follows the central idea of combining an encoder and decoder with a
Vision Mamba bottleneck and hypergraph-based reasoning. The model
variants differ in their encoder, hypergraph construction, and use of
symbolic anatomical constraints.

**Research focus:** produce wound masks with accurate region overlap and
boundaries while encouraging anatomically consistent predictions.

> **Important:** This repository is a research prototype, not a medical
> device. Outputs are intended for research and must not be used as the
> sole basis for diagnosis, treatment, or clinical decisions.

## Key Features

-   **Vision Mamba context modelling:** uses a selective state-space
    bottleneck to model long-range image context.
-   **Hypergraph reasoning:** represents higher-order relationships
    among multiple image nodes rather than only pairwise connections.
-   **Anatomical constraints:** selected variants incorporate
    relationships between the wound bed, wound edge, surrounding tissue,
    and healthy skin.
-   **Four architectural variants:** fixed anatomical hyperedges,
    dynamically constructed hyperedges, learnable region-wise fusion,
    and hierarchical multi-scale hypergraphs.
-   **Segmentation evaluation:** reports Dice, Intersection over Union
    (IoU), precision, recall, and 95th-percentile Hausdorff distance
    (HD95).
-   **Notebook workflow:** training, evaluation, and visual analysis are
    provided in Jupyter notebooks.

## Architecture at a Glance

The common processing pipeline is:

1.  **Input and preprocessing:** crop around the wound, resize the
    image, normalise image channels, and augment training examples.
2.  **Feature encoder:** extract visual features using the backbone
    selected for the model variant.
3.  **Vision Mamba bottleneck:** model long-range spatial context using
    a selective state-space module.
4.  **Hypergraph reasoning:** pass information among groups of related
    nodes.
5.  **Symbolic anatomical fusion:** apply anatomical constraints in the
    variants that implement them.
6.  **Segmentation decoder:** reconstruct a pixel-level wound mask.
7.  **Evaluation:** compare predicted masks with ground-truth masks
    using overlap and boundary metrics.

## Model Variants

  ------------------------------------------------------------------------------------------------------------------------------
  Notebook                                                               Main design      Hypergraph           Input size in the
                                                                                          strategy                         paper
  ---------------------------------------------------------------------- ---------------- ---------------- ---------------------
  `ResNet50_VisionMamba_Hypergraph_Segmentation.ipynb`                   ResNet50 encoder Fixed anatomical             512 × 512
                                                                         with Vision      hyperedges and   
                                                                         Mamba            hyperedge        
                                                                                          attention        

  `ConvNeXt_Mamba_DynamicHypergraph_Segmentation.ipynb`                  ConvNeXt Base    Dynamic                      512 × 512
                                                                         encoder with     hyperedges based 
                                                                         Vision Mamba     on feature       
                                                                                          similarity       

  `UNet_Mamba_SymbolicHypergraph_Segmentation.ipynb`                     Three-stage      Visual and                   384 × 384
                                                                         U-Net with       symbolic         
                                                                         Vision Mamba     information with 
                                                                                          a learnable      
                                                                                          fusion gate      

  `ConvNeXt_VisionMamba_MultigranularityHypergraph_Segmentation.ipynb`   ConvNeXt and     Hierarchical                 512 × 512
                                                                         Vision Mamba     hypergraphs at   
                                                                                          fine, middle,    
                                                                                          and coarse       
                                                                                          scales           
  ------------------------------------------------------------------------------------------------------------------------------

The model names describe the notebook implementations. Check each
notebook's configuration cells before running, because parameters and
data-handling details may differ.

## Reported Results

The accompanying research paper reports the following test-set results.
Segmentation metrics were calculated on the 168 wound-positive images
among the 229 test images; images without wounds were assessed
separately.

  ------------------------------------------------------------------------------
  Model               Dice ↑        IoU ↑  Precision ↑     Recall ↑       HD95 ↓
                                                                        (pixels)
  ------------- ------------ ------------ ------------ ------------ ------------
  Model 1:            0.9365       0.8838   **0.9412**       0.9364    **21.71**
  ResNet50 +                                                        
  fixed                                                             
  hypergraph                                                        

  Model 2:        **0.9367**   **0.8843**       0.9341       0.9451        23.43
  ConvNeXt +                                                        
  dynamic                                                           
  hypergraph                                                        

  Model 3:            0.8759       0.7960       0.8748       0.9001        35.21
  U-Net +                                                           
  learnable                                                         
  fusion                                                            

  Model 4:            0.9007       0.8305       0.8558   **0.9703**        28.69
  Multi-scale                                                       
  hypergraph                                                        
  ------------------------------------------------------------------------------

**Metric interpretation** - **Dice** and **IoU** measure overlap between
the predicted and ground-truth masks. Higher is better. - **Precision**
measures how much of the predicted wound region is correct. Higher is
better. - **Recall** measures how much of the ground-truth wound region
is detected. Higher is better. - **HD95** measures boundary discrepancy
using the 95th percentile of Hausdorff distances. Lower is better.

According to the paper, Model 2 achieved the highest Dice and IoU, Model
1 achieved the highest precision and lowest HD95, and Model 4 achieved
the highest recall. These are results reported by the paper and are not
a guarantee of performance on other datasets or clinical settings.

## Dataset

The study uses a combined set of **1,370 clinical images** from:

-   **Foot Ulcer Segmentation Challenge (FUSC)**
-   **Medetec Foot Ulcer dataset**

The paper describes a stratified **70:15:15** split:

  Split             Images
  ------------ -----------
  Training             913
  Validation           228
  Test                 229
  **Total**      **1,370**

Preprocessing described in the paper includes wound-region cropping,
resizing, and ImageNet channel normalisation. Training augmentations
include horizontal and vertical flips, 90-degree rotations, elastic
deformation, and colour jitter, with the same spatial transformations
applied to images and masks.

### Dataset access and layout

The notebooks currently expect the dataset under this Kaggle path:

``` text
/kaggle/input/datasets/adnanjan01/wound-seg-3-datasets/data
```

The code searches beneath this directory for folders corresponding to
the Foot Ulcer and Medetec datasets, and expects split folders with
image and label directories. A typical layout is:

``` text
data/
├── Foot Ulcer.../
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── validation/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── Medetec.../
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

Actual folder names and available splits can vary by dataset release.
Inspect the dataset and the notebook's data-loading cell before
training. Obtain the datasets from their official sources and follow
their licences, access conditions, and research-use requirements. The
datasets are not bundled with this repository.

## Repository Structure

``` text
wound-segmentation/
├── README.md
├── ResNet50_VisionMamba_Hypergraph_Segmentation.ipynb
├── ConvNeXt_Mamba_DynamicHypergraph_Segmentation.ipynb
├── UNet_Mamba_SymbolicHypergraph_Segmentation.ipynb
└── ConvNeXt_VisionMamba_MultigranularityHypergraph_Segmentation.ipynb
```

## Environment and Dependencies

The notebooks use Python and PyTorch, along with computer-vision and
scientific-computing libraries. The imports include:

-   PyTorch and Torchvision
-   NumPy
-   Pillow
-   OpenCV
-   Matplotlib
-   scikit-learn
-   SciPy

A CUDA-capable GPU is recommended for training, especially for the 512 ×
512 models. CPU execution may be possible but can be significantly
slower. Exact compatible versions are not pinned in the provided
notebooks, so install versions compatible with your Python, PyTorch,
Torchvision, CUDA, and GPU environment.

Example environment setup:

``` bash
python -m venv .venv
```

Activate the environment, then install the required packages:

``` bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow opencv-python matplotlib scikit-learn scipy jupyter
```

For GPU training, follow the official PyTorch installation selector to
install a build compatible with your CUDA setup:
https://pytorch.org/get-started/locally/

## Getting Started

### Option A: Run on Kaggle

The notebook configuration uses a Kaggle dataset path, so Kaggle is the
most direct starting point.

1.  Upload or import this repository's notebooks into a Kaggle notebook
    environment.
2.  Attach the required FUSC and Medetec dataset files to the notebook.
3.  Confirm that the dataset is mounted at the path expected by the
    notebook, or update `DATA_BASE` in the configuration cell.
4.  Enable a GPU accelerator if available.
5.  Open one of the model notebooks and run the cells in order.
6.  Review the printed dataset paths and sample visualisations before
    starting training.
7.  Inspect validation metrics, test metrics, and predicted-mask
    visualisations after execution.

### Option B: Run Locally

1.  Clone or download the repository.

2.  Install the dependencies listed above.

3.  Download and arrange the datasets according to the expected folder
    structure.

4.  Open the selected notebook:

    ``` bash
    jupyter lab
    ```

5.  Update `DATA_BASE` in the notebook to the local dataset directory.

6.  Run the notebook cells in order.

**Before training:** confirm that image files and their corresponding
masks are paired correctly, that masks are loaded as segmentation
labels, and that the dataset split does not introduce leakage. Notebook
execution time and memory use depend on the model, hardware, and
dataset.

## Training Configuration

The notebooks use configurations based on the model variant. Common
settings described in the paper include:

  Setting                 Paper configuration
  ----------------------- ----------------------------------
  Optimiser               AdamW
  Initial learning rate   0.0001
  Weight decay            0.0001
  Maximum epochs          50
  Warm-up                 5 epochs
  Batch size              4 in the notebook configurations
  Gradient clipping       Maximum norm of 1.0
  Model selection         Combined validation Dice and IoU
  Inference               Six-fold test-time augmentation
  Random seed             42 in the notebooks

The training schedule uses linear warm-up followed by cosine annealing.
The paper also describes early stopping and exponential moving average
(EMA) weights. Individual notebook implementations may expose additional
settings. Check the relevant notebook before changing the configuration
or interpreting a run.

## Evaluation

The project evaluates segmentation using Dice, IoU, precision, recall,
and HD95. It also includes visual analysis of input images, ground-truth
masks, predicted masks, and error maps in the notebooks.

When reproducing the reported results:

1.  Use the same dataset version and split where available.
2.  Keep preprocessing and image-mask transformations aligned.
3.  Select checkpoints using validation data only.
4.  Apply the same test-time augmentation procedure if comparing against
    the paper's reported test results.
5.  Report all relevant metrics, not only the best score.
6.  Clearly distinguish results reproduced locally from the values
    reported in the paper.

## Reproducibility Notes and Limitations

-   The published results use one combined FUSC and Medetec dataset.
    Independent external clinical validation is still needed.
-   Six-fold test-time augmentation adds inference cost and may limit
    real-time deployment.
-   The four architectures have different strengths. High recall does
    not necessarily imply high precision or the best boundary accuracy.
-   The reported metrics should not be interpreted as evidence of
    clinical readiness.
-   Dataset versions, label conventions, package versions, hardware, and
    random seeds can affect results.
-   The notebooks are the source of truth for implementation details.
    This README does not claim that a separate command-line training or
    inference interface is provided.

## Citation

If you use this work in academic research, please cite the associated
paper. Replace or supplement this entry with the final publication
details when available.

``` text
Ahmad, S., Iqbal, H., Farooq, J. A., and Panda, G.
"Neurosymbolic Vision Mamba with Hypergraph Reasoning for Anatomically
Consistent Diabetic Foot Ulcer Segmentation."
C.V. Raman Global University.
```

## Authors

-   Shayam Ahmad
-   Haroon Iqbal
-   Jan Adnan Farooq
-   Ganapati Panda

**Affiliation:** C.V. Raman Global University

## Responsible Use

This project is intended for academic research and technical
experimentation. It has not been established as a clinically validated
diagnostic system. Do not use its predictions as a substitute for
professional medical assessment. Any clinical application would require
appropriate external validation, risk assessment, regulatory review, and
oversight by qualified healthcare professionals.

## Acknowledgements

The work uses the Foot Ulcer Segmentation Challenge (FUSC) and Medetec
foot-ulcer image datasets. Please acknowledge the dataset creators and
follow their respective terms of use when using the data.
