<p align="center">
  <img src="docs/assets/banner.svg" alt="OCTNet retinal OCT classification banner" width="100%">
</p>

<h1 align="center">OCTNet: Retinal OCT Image Classification</h1>

<p align="center">A lightweight ConvNeXt-inspired CNN with SE attention for eight-class retinal OCT image classification.</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#method">Method</a> ·
  <a href="#reproducibility">Reproducibility</a>
</p>

> **Medical disclaimer:** This repository is a course project for research and education. It is not a medical device and must not be used for diagnosis or clinical decision-making.

## Project Overview

OCTNet classifies retinal optical coherence tomography (OCT) images into eight categories: `AMD`, `CNV`, `CSR`, `DME`, `DR`, `DRUSEN`, `MH`, and `NORMAL`. The project turns the original course project into a reproducible PyTorch training, evaluation, and single-image inference pipeline.

## Highlights

- **Compact custom model:** 3.60M parameters and 2.32G FLOPs for a 224×224 input.
- **Modern CNN components:** 7×7 depthwise convolution, GroupNorm, residual connections, and SE channel attention.
- **Robust training recipe:** label smoothing, AdamW, cosine learning-rate decay, mixed precision, and four image augmentation operations.
- **Complete evaluation:** Accuracy, per-class precision/recall/F1, and an annotated confusion matrix.
- **Runnable workflow:** separate `train.py`, `evaluate.py`, and `predict.py` entry points with YAML configuration and saved checkpoints.

## Results

The figures and metrics below are preserved from the original experiment on the provided held-out test split (2,800 images; 350 per class).

| Metric | Value |
| --- | ---: |
| Best validation Accuracy | **96.64%** (epoch 15) |
| Test Accuracy | **97.04%** |
| Macro F1-score | **97.03%** |
| Weighted F1-score | **97.03%** |

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| AMD | 1.0000 | 1.0000 | 1.0000 |
| CNV | 0.9371 | 0.9371 | 0.9371 |
| CSR | 1.0000 | 1.0000 | 1.0000 |
| DME | 0.9528 | 0.9229 | 0.9376 |
| DR | 1.0000 | 1.0000 | 1.0000 |
| DRUSEN | 0.9419 | 0.9257 | 0.9337 |
| MH | 1.0000 | 1.0000 | 1.0000 |
| NORMAL | 0.9319 | 0.9771 | 0.9540 |

<p align="center">
  <img src="figures/training_curves.png" alt="Training and validation curves" width="95%">
</p>

<p align="center">
  <img src="figures/confusion_matrix.png" alt="Test-set confusion matrix" width="72%">
</p>

The main errors occur among `CNV`, `DME`, `DRUSEN`, and `NORMAL`, whose OCT appearances can share fluid, elevation, or texture patterns. See [`docs/results.md`](docs/results.md) for a concise analysis.

## Method

OCTNet retains the original experimental architecture. It is **ConvNeXt-inspired**, rather than an official ConvNeXt implementation.

```mermaid
flowchart LR
    A[224×224 RGB OCT image] --> B[Stage 1: 3→32]
    B --> C[Stage 2: 32→64]
    C --> D[Stage 3: 64→128 + SE]
    D --> E[Stage 4: 128→256 + SE]
    E --> F[Stage 5: 256→512 + SE]
    F --> G[Global Average Pooling]
    G --> H[512 → 1024 → 8 classifier]
```

Each stage applies a channel projection when needed, a residual 7×7 depthwise convolution block (`GroupNorm → 1×1 expansion → GELU → 1×1 projection`), an optional SE block, and 2×2 max pooling. Detailed design notes are available in [`docs/method.md`](docs/method.md).

### Training Recipe

| Component | Setting |
| --- | --- |
| Input | 224×224, bicubic resize |
| Normalization | mean/std = `[0.210, 0.210, 0.210]` / `[0.182, 0.182, 0.182]` |
| Augmentation | horizontal flip, ±15° rotation, brightness/contrast jitter, RandomErasing (`p=0.4`) |
| Optimizer | AdamW, lr = `5e-4`, weight decay = `3e-5` |
| Loss | CrossEntropyLoss with label smoothing = `0.1` |
| Scheduler | CosineAnnealingLR, `T_max=20`, `eta_min=1e-6` |
| Batch size / epochs | 64 / 20 |
| Model selection | highest validation Accuracy |

## Dataset

This project uses the [Retinal OCT C8 dataset on Kaggle](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8/versions/2), which contains 24,000 OCT images across eight classes. The dataset is **not** redistributed in this repository.

Prepare the folder structure below, keeping the class names consistent across all splits:

```text
data/RetinalOCT_Dataset/
├── train/  # 18,400 images
├── val/    # 2,800 images
└── test/   # 2,800 images
```

Each split contains one subdirectory per class. More detail is in [`dataset/README.md`](dataset/README.md). Please comply with the data source's license and terms before downloading or redistributing it.

## Quick Start

### 1. Create an environment

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for your CUDA environment first, then install the remaining dependencies:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd retinal-oct-classification

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate retinal-oct-classification
```

### 2. Download data

Download and extract the dataset into `data/RetinalOCT_Dataset/`, or point the commands below to another location using `--data-root`.

### 3. Train

```bash
python train.py --config configs/octnet.yaml
```

This writes `best.pt`, `history.json`, `training_curves.png`, the effective configuration, and logs to `outputs/train/` by default.

### 4. Evaluate

```bash
python evaluate.py \
  --checkpoint outputs/train/best.pt \
  --split test \
  --output-dir outputs/evaluation
```

The command saves `classification_report.txt` and `confusion_matrix.png`.

### 5. Predict one image

```bash
python predict.py \
  --checkpoint outputs/train/best.pt \
  --image path/to/oct_image.png \
  --top-k 3
```

The command prints top-k class probabilities as JSON. No dataset directory is needed for single-image inference.

## Reproducibility

- `configs/octnet.yaml` centralizes hyperparameters and preprocessing values.
- `utils/seed.py` seeds Python, NumPy, and PyTorch; a run's configuration and best checkpoint are stored under its output directory.
- `train.py` correctly serializes the best validation-Accuracy checkpoint at the time it is observed, rather than holding a mutable state-dict reference.
- Exact equality with the original notebook's numbers cannot be guaranteed: the original run did not record a random seed, dependency versions, or its trained weights. The displayed results are retained as reported artifacts.

See [`docs/reproducibility.md`](docs/reproducibility.md) for platform notes.

## Repository Layout

```text
.
├── configs/          # Training configuration
├── dataset/          # Dataset placement instructions
├── docs/             # Method, results, reproducibility, and assets
├── examples/         # Command examples
├── figures/          # Versioned result figures
├── models/           # OCTNet and building blocks
├── notebooks/        # Archived original course notebook
├── utils/            # Data, training, metrics, logging, and visualization
├── train.py          # Training and validation entry point
├── evaluate.py       # Labeled-split evaluation entry point
└── predict.py        # Single-image inference entry point
```

## Limitations and TODO

- [ ] Publish a verified release checkpoint with exact environment metadata.
- [ ] Add a class-wise error gallery and Grad-CAM visualization.
- [ ] Add baseline and attention ablations.
- [ ] Validate on an external, patient-disjoint test set.

The current experiment is limited to one dataset split. It does not establish clinical utility, patient-level generalization, or causality of individual design choices.

## Citation

If this repository is useful in your work, please cite it after replacing the author and repository URL with your published information:

```bibtex
@misc{wang2026octnet,
  title  = {OCTNet: Retinal OCT Image Classification},
  author = {Jianjie Wang},
  year   = {2026},
  howpublished = {\url{<YOUR_GITHUB_REPOSITORY_URL>}}
}
```

## Acknowledgements

- The Retinal OCT C8 dataset and its contributors.
- [ConvNeXt](https://arxiv.org/abs/2201.03545), [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), [label smoothing](https://arxiv.org/abs/1512.00567), and [Random Erasing](https://arxiv.org/abs/1708.04896).
