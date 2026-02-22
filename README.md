# 🏜️ Perception Engine — Offroad Semantic Segmentation

> **Duality AI Hackathon Submission** — Autonomous UGV terrain analysis using DINOv2 + 4D Bayesian Fusion

<p align="center">
  <img src="https://img.shields.io/badge/mIoU-61%25-brightgreen?style=for-the-badge" alt="mIoU">
  <img src="https://img.shields.io/badge/Pixel_Accuracy-81.2%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/Classes-10-blue?style=for-the-badge" alt="Classes">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" alt="PyTorch">
</p>

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Interactive UI](#-interactive-ui)
- [Scripts & Utilities](#-scripts--utilities)
- [Development](#-development)
- [Documentation](#-documentation)

---

## 📋 Overview

A complete perception pipeline for **offroad semantic segmentation** of desert terrain, designed for autonomous UGV (Unmanned Ground Vehicle) navigation. The system assigns one of 10 terrain classes to every pixel in an input image, generates traversability cost maps, and computes safe navigation paths.

### Key Results

| Metric | Value |
|---|---|
| **Highest mIoU** | **61%** (ensemble) |
| **V6 Single-Model mIoU** | 52.11% |
| **Pixel Accuracy** | 81.2% |
| **Test Set** | 1,002 images (960×540) |
| **Classes** | 10 terrain types |

### Top Per-Class IoU (1,002-image test set)

| Class | IoU |
|---|---|
| Sky | 99.0% |
| Landscape | 72.7% |
| Tree | 59.8% |
| Dry Bush | 45.5% |
| Dry Grass | 44.3% |
| Rock | 43.2% |

---

## 🏗️ Architecture

### Why DINOv2?

Standard CNNs (ResNet-based DeepLabV3+) achieved only **16–17% mIoU** due to inherent **texture bias** — they memorized synthetic textures that don't transfer to unseen desert environments. We switched to **DINOv2 ViT-S/14**, a Vision Transformer trained via self-supervised learning on 142M images, which learns **domain-agnostic structural features** instead of texture patterns.

### Model Architecture

```
Input (960×540 RGB)
    │
    ▼
DINOv2 ViT-S/14 (frozen, 21M params)
    │  14×14 patches → 34×19 = 646 tokens × 384-dim
    ▼
Custom ConvNeXt Segmentation Head (~2.1M params)
    ├── Block 1: 7×7 depthwise sep. conv (broad context)
    ├── Block 2: 5×5 depthwise sep. conv (medium detail)
    ├── Block 3: 3×3 depthwise sep. conv (local boundaries)
    └── 1×1 conv → 10 class logits
    │
    ▼
Bilinear upsample to 960×540
    │
    ▼
4D Bayesian Fusion: P(class | H, S, V, Y)
    │  Multiplicative fusion with 25,920 bins/class histogram
    ▼
Final Segmentation Mask (10 classes)
```

### 4D Bayesian Fusion (Key Innovation)

The single largest source of error was **rock vs landscape confusion** — they share identical HSV color profiles. Our solution:

1. **Build** a 4D Joint Histogram `P(class | Hue, Saturation, Value, Y-position)` from training data (18×12×12×10 = 25,920 bins per class)
2. **Fuse** during inference: `P_final ∝ P_DNN^0.40 × P_histogram^0.55 × W_freq`
3. The histogram encodes environment-invariant spatial constraints (sky=top, rocks=elevated, landscape=flat ground)

This alone boosted mIoU by **+8.5%** over the neural network alone.

---

## 📊 Version History

Our development followed a rigorous iterative process. Each version addressed a specific bottleneck identified through quantitative analysis:

| Version | Architecture | mIoU | Key Change |
|---|---|---|---|
| V1 | ResNet34 DeepLabV3+ | 16% | Baseline CNN |
| V2 | ResNet50 DeepLabV3+ | 17% | Deeper backbone (still texture-biased) |
| V3 | DINOv2 ViT-S/14 | 35% | Foundation model (+106% jump) |
| V4 | V3 + Freq. Recalibration | 45% | Class rebalancing (+29%) |
| V5 | V4 + Multi-scale + 1D Prior | 48% | Spatial context (+7%) |
| **V6** | **V5 + 4D Bayesian Fusion** | **52.1%** | **Joint histogram (+8.5%)** |
| **Ensemble** | **V3+V5+V6 + Prior** | **61%** | **Multi-model fusion** |

---

## 📁 Project Structure

```
Perception/
├── perception_engine/                  # Core Python package
│   ├── api/                            # FastAPI backend & inference pipeline
│   │   ├── server.py                   # REST API server
│   │   └── pipeline.py                 # Runs infer_ensemble.py, returns base64 outputs
│   ├── configs/                        # Configuration management
│   │   ├── config_loader.py            # YAML loading, validation, device resolution
│   │   ├── experiment.yaml             # Main experiment config (models, costs, planner)
│   │   └── config_v*.yaml              # Model-specific configs (v2, v3, v5, v6)
│   ├── core/                           # Shared data structures
│   │   └── data_types.py               # SegmentationOutput, BenchmarkReport, NavigationResult
│   ├── engine/                         # Inference & preprocessing
│   │   ├── inference_engine.py         # Batch inference orchestration
│   │   ├── mask_remapping.py           # Raw mask values → class indices
│   │   ├── postprocessing.py           # CRF, morphological ops
│   │   └── preprocessing.py           # Image normalization, resizing
│   ├── evaluation/                     # Metrics & benchmarking
│   │   ├── segmentation_metrics.py     # mIoU, fwIoU, Dice, confusion matrix
│   │   ├── calibration.py              # ECE, reliability diagrams
│   │   ├── robustness.py               # Perturbation testing
│   │   ├── benchmarking.py             # Multi-model benchmark runner
│   │   ├── batch_runner.py             # Batch evaluation over dataset
│   │   ├── export.py                   # JSON/CSV report export
│   │   └── multiscale_robustness.py     # Scale-invariance evaluation
│   ├── models/                         # Model registry & loading
│   │   ├── base_model.py               # Abstract model interface
│   │   ├── registry.py                 # Model factory
│   │   └── loaders.py                  # Weight loading utilities
│   ├── navigation/                     # Path planning & safety
│   │   ├── cost_mapping.py             # Segmentation → traversability cost grid
│   │   ├── planner.py                  # A* path planning
│   │   └── safety.py                   # Safety scoring
│   ├── training/                       # Training & inference scripts
│   │   ├── train_dinov2_v6_kaggle.py   # V6 training (Kaggle dual T4)
│   │   ├── train_dinov2_v5_kaggle.py   # V5 training
│   │   ├── train_v3_kaggle.py          # V3 training
│   │   ├── build_joint_histograms.py   # Build 4D prior from training data
│   │   ├── eval_batch.py               # Full 1,002-image evaluation
│   │   ├── infer_ensemble.py           # Ensemble inference + viz (primary CLI)
│   │   ├── infer_dinov2.py             # Single-model (V6) inference
│   │   └── inference_v5.py             # V5 single-model inference
│   ├── tests/                          # Unit & integration tests
│   ├── visualization/                  # Overlay, confidence maps
│   │   └── overlays.py
│   └── Offroad_Segmentation_testImages/# Test dataset (gitignored, ~1.1GB)
│       ├── Color_Images/               # 1,002 RGB inputs (960×540)
│       └── Segmentation/               # Ground truth masks
├── weights/                            # Model weights & priors (gitignored)
│   ├── best_model_v*.pth               # Trained checkpoints
│   └── joint_histograms.pkl            # 4D Bayesian prior
├── scripts/                            # Utility scripts
│   ├── run_perception_api.sh           # Start API server (uvicorn)
│   ├── test_pipeline.py                # Quick pipeline smoke test
│   └── offroad_segmentation/           # Dataset utilities
│       ├── test_segmentation.py        # Validation/evaluation script
│       ├── visualize.py               # Colorize segmentation masks
│       └── ENV_SETUP/                   # Windows env setup scripts
├── docs/
│   └── README_PERCEPTION_API.md        # API documentation
├── ui/                                 # Next.js frontend
│   └── app/(dashboard)/
│       ├── page.tsx                     # Root → redirects to /technical
│       ├── technical/page.tsx          # Command Center (evaluation dashboard)
│       ├── perception/page.tsx         # Perception Lab (upload & infer)
│       ├── arena/page.tsx              # Model Arena
│       ├── inference/page.tsx          # Inference playground
│       ├── comparison/page.tsx          # Model comparison
│       ├── config/page.tsx             # Experiment config
│       ├── simulator/page.tsx          # Simulator
│       └── settings/page.tsx           # Settings
├── pyproject.toml                      # Python package config
└── README.md                            # This file
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.9+ |
| **PyTorch** | 1.12+ (2.0+ recommended) |
| **Node.js** | 18+ (for UI) |

### 1. Install Python Dependencies

```bash
# From project root
pip install -e .

# For API + UI: install optional API deps
pip install ".[api]"
```

### 2. Download Model Weights & Priors

Download from Google Drive and place in the **`weights/`** directory:

📦 **[Download Models & Priors (Google Drive)](https://drive.google.com/drive/folders/1-Hr0p6j26F9GVr5ISqdxa32J5zSJXtHa?usp=sharing)**

| File | Size | Required |
|---|---|---|
| `best_model_v6.pth` | 200 MB | ✅ Essential |
| `joint_histograms.pkl` | 2 MB | ✅ Essential |
| `best_model_v3.pth` | 102 MB | For ensemble (61% mIoU) |
| `best_model_v5.pth` | 146 MB | For ensemble (61% mIoU) |

### 3. Download Test Dataset (Optional)

For full evaluation, download the test images and place them in:
- `perception_engine/Offroad_Segmentation_testImages/Color_Images/`
- `perception_engine/Offroad_Segmentation_testImages/Segmentation/`

### 4. Run Single-Image Inference (CLI)

```bash
python perception_engine/training/infer_ensemble.py \
  --image perception_engine/Offroad_Segmentation_testImages/Color_Images/0000096.png \
  --gt perception_engine/Offroad_Segmentation_testImages/Segmentation/0000096.png \
  --v6-weights weights/best_model_v6.pth \
  --output outputs/
```

**Outputs:**
- `ensemble_0000096_overlay.png` — Segmentation overlay on input
- `ensemble_0000096_comparison.png` — Side-by-side with ground truth
- Console: mIoU, per-class IoU, pixel accuracy

### 5. Run Full Test Set Evaluation

```bash
python perception_engine/training/eval_batch.py \
  --weights weights/best_model_v6.pth
```

### 6. Run the Interactive UI

```bash
# Terminal 1: Backend API
PERCEPTION_WEIGHTS=weights/best_model_v6.pth \
PERCEPTION_PRIOR=weights/joint_histograms.pkl \
python -m uvicorn perception_engine.api.server:app --port 8000

# Or use the convenience script:
./scripts/run_perception_api.sh

# Terminal 2: Frontend
cd ui && npm install && npm run dev
```

Open **http://localhost:3000**:
- **Perception Lab** (`/perception`) → Upload image, run inference, view overlay + cost map + path
- **Command Center** (`/technical`) → Full evaluation: version progression, per-class IoU, robustness, confusion matrix

---

## ⚙️ Configuration

### Experiment Config (`perception_engine/configs/experiment.yaml`)

The main config drives the evaluation pipeline:

- **`models`** — List of model definitions (architecture, backbone, weights path)
- **`class_names`** — 10 terrain classes (tree, lush_bush, dry_grass, …)
- **`mask_value_mapping`** — Raw mask pixel values → contiguous indices
- **`cost_mapping`** — Class IDs → traversable / obstacle / soft / ignored
- **`planner`** — A* settings (start, goal, diagonal moves)
- **`preprocessing`** — Resize, normalization
- **`robustness`** — Perturbation tests (brightness, blur, noise)
- **`output`** — Output directory, save visualizations

### Model Configs (`perception_engine/configs/config_v*.yaml`)

Model-specific configs (input size, token grid, preprocessing) are saved during training. Place `config_v5.yaml`, `config_v6.yaml` in `perception_engine/configs/` for single-model inference (e.g. `inference_v5.py`).

### Config Loader

```python
from perception_engine.configs.config_loader import load_config, get_device

config = load_config("perception_engine/configs/experiment.yaml")
device = get_device(config)  # "cuda" | "mps" | "cpu"
```

---

## 🔬 Training

### Training on Kaggle (Dual T4 GPUs)

```bash
# Upload train_dinov2_v6_kaggle.py to Kaggle Notebook
# Set GPU accelerator: T4 × 2
python train_dinov2_v6_kaggle.py
```

**Training Configuration:**
- Loss: Cross-Entropy with inverse-frequency class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: Cosine Annealing with warm restarts
- Duration: 50 epochs
- Backbone: Frozen (only ~2.1M-param head trained)

### Building the 4D Prior

```bash
python perception_engine/training/build_joint_histograms.py \
  --train-dir <path_to_training_data>
```

Extracts `P(class | H, S, V, Y)` from training images → `joint_histograms.pkl`.

---

## 🧪 Evaluation

### Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection over Union (primary) |
| **Pixel Accuracy** | % correctly classified pixels |
| **Per-Class IoU** | IoU per terrain class |
| **Confusion Matrix** | Pixel-level misclassification |

### Benchmark Runner (Experiment Mode)

```bash
python -m perception_engine.run_experiment \
  --config perception_engine/configs/experiment.yaml \
  --image path/to/image.png \
  --ground-truth path/to/gt.png
```

```bash
# Batch mode (uses image_dir/gt_dir from config)
python -m perception_engine.run_experiment \
  --config perception_engine/configs/experiment.yaml \
  --batch --max-samples 10
```

### Top Confusion Pairs

| Ground Truth → Prediction | % of Total | Root Cause |
|---|---|---|
| dry_grass → rock | 6.29% | Identical HSV color profiles |
| landscape → rock | 6.16% | Color overlap + adjacent positions |
| rock → landscape | 3.10% | Reverse confusion at boundaries |

### Limitations

- **Rock vs Landscape**: Share identical color; mitigated by spatial prior; breaks down in flat boulder fields
- **Absent Classes**: `ground_clutter`, `flower`, `log` had zero training samples
- **Sparse Classes**: `lush_bush` (15K pixels) → insufficient data

---

## 🖥️ Interactive UI

| Page | Route | Description |
|---|---|---|
| **Command Center** | `/technical` | Evaluation dashboard: version progression, per-class IoU, confusion matrix, robustness |
| **Perception Lab** | `/perception` | Upload image → run ensemble inference → overlay, cost map, A* path |
| **Model Arena** | `/arena` | Compare models side-by-side |
| **Inference** | `/inference` | Inference playground |
| **Comparison** | `/comparison` | Model comparison |
| **Config** | `/config` | Experiment config viewer |
| **Simulator** | `/simulator` | Simulator |
| **Settings** | `/settings` | App settings |

The Perception Lab runs the **exact same** `infer_ensemble.py` CLI as a subprocess — UI output matches CLI output.

---

## 🛠️ Scripts & Utilities

| Script | Purpose |
|---|---|
| `scripts/run_perception_api.sh` | Start API server (`uvicorn` on port 8000) |
| `scripts/test_pipeline.py` | Smoke test: run pipeline on sample image |
| `scripts/offroad_segmentation/test_segmentation.py` | Validation on test set |
| `scripts/offroad_segmentation/visualize.py` | Colorize segmentation masks |

**Quick pipeline test:**
```bash
python scripts/test_pipeline.py
```

---

## 🛡️ Generalization Strategy

1. **Domain-Invariant Features**: Frozen DINOv2 learns structure, not texture (17% → 35% mIoU)
2. **4D Bayesian Prior**: Spatial constraints (sky=top, rocks=elevated) transfer across environments
3. **Class Frequency Recalibration**: Boosts underrepresented obstacle classes (rock: 3.5×)

---

## 🧑‍💻 Development

### Run Tests

```bash
# All tests
pytest perception_engine/tests/ -v

# With coverage
pytest perception_engine/tests/ -v --cov=perception_engine

# Skip e2e/integration (require weights)
pytest perception_engine/tests/ -v --ignore=perception_engine/tests/test_end_to_end.py --ignore=perception_engine/tests/test_integration.py
```

### Project Layout

- **`perception_engine/core/`** — Data types used across the pipeline
- **`perception_engine/configs/`** — Config loading & model configs
- **`perception_engine/api/`** — FastAPI server; pipeline wraps `infer_ensemble.py`
- **`perception_engine/evaluation/`** — Metrics, benchmarking, export
- **`perception_engine/training/`** — Training scripts (Kaggle) and inference CLI scripts

---

## 📚 Documentation

- **[API Documentation](docs/README_PERCEPTION_API.md)** — FastAPI setup, endpoints, environment variables

---

## 📄 License

This project was developed for the Duality AI Hackathon. MIT License.

---

## 👥 Team

Developed as part of the perception challenge for autonomous UGV terrain analysis.
