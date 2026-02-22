# 🏜️ Perception Engine — Offroad Semantic Segmentation

> **Duality AI Hackathon Submission** — Autonomous UGV terrain analysis using DINOv2 + 4D Bayesian Fusion

<p align="center">
  <img src="https://img.shields.io/badge/mIoU-61%25-brightgreen?style=for-the-badge" alt="mIoU">
  <img src="https://img.shields.io/badge/Pixel_Accuracy-81.2%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/Classes-10-blue?style=for-the-badge" alt="Classes">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" alt="PyTorch">
</p>

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
Monospace/
├── perception_engine/                  # Core Python package
│   ├── training/
│   │   ├── train_dinov2_v6_kaggle.py   # V6 training script (Kaggle dual T4)
│   │   ├── build_joint_histograms.py   # Extract 4D priors from training data
│   │   ├── eval_batch.py               # Full 1,002-image evaluation
│   │   ├── infer_ensemble.py           # Single-image ensemble inference + viz
│   │   └── infer_dinov2.py             # Single-model inference
│   ├── api/
│   │   ├── server.py                   # FastAPI backend
│   │   └── pipeline.py                 # Inference pipeline (calls infer_ensemble.py)
│   ├── engine/
│   │   ├── inference_engine.py         # Batch inference engine
│   │   ├── mask_remapping.py           # Class ID mapping
│   │   └── postprocessing.py           # CRF, morphological ops
│   ├── evaluation/
│   │   ├── segmentation_metrics.py     # mIoU, fwIoU, Dice
│   │   ├── calibration.py              # ECE, reliability diagrams
│   │   ├── robustness.py               # Perturbation tests
│   │   └── benchmarking.py             # Automated benchmark runner
│   ├── models/
│   │   ├── base_model.py               # Abstract model interface
│   │   └── registry.py                 # Model factory
│   ├── navigation/
│   │   ├── cost_mapping.py             # Traversability cost grid
│   │   ├── planner.py                  # A* path planning
│   │   └── safety.py                   # Safety scoring
│   ├── tests/                          # Unit & integration tests
│   └── Offroad_Segmentation_testImages/# Test dataset
│       ├── Color_Images/               # 1,002 RGB inputs (960×540)
│       └── Segmentation/               # Ground truth masks
├── ui/                                 # Next.js frontend
│   └── app/(dashboard)/
│       ├── page.tsx                     # Command Center dashboard
│       ├── perception/page.tsx          # Perception Lab (interactive inference)
│       └── technical/page.tsx           # Technical evaluation page
├── pyproject.toml                       # Python package config
└── README.md                            # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Node.js 18+ (for UI)

### 1. Install Python Dependencies

```bash
pip install -e .
```

### 2. Download Model Weights

Download the trained weights and priors from Google Drive and place them in the project root:

📦 **[Download Models & Priors (Google Drive)](https://drive.google.com/drive/folders/1-Hr0p6j26F9GVr5ISqdxa32J5zSJXtHa?usp=sharing)**

| File | Size | Required |
|---|---|---|
| `best_model_v6.pth` | 200 MB | ✅ Essential |
| `joint_histograms.pkl` | 2 MB | ✅ Essential |
| `best_model_v3.pth` | 102 MB | For ensemble (61% mIoU) |
| `best_model_v5.pth` | 146 MB | For ensemble (61% mIoU) |

### 3. Run Single-Image Inference (CLI)

```bash
python perception_engine/training/infer_ensemble.py \
  --image perception_engine/Offroad_Segmentation_testImages/Color_Images/0000096.png \
  --gt perception_engine/Offroad_Segmentation_testImages/Segmentation/0000096.png \
  --v6-weights best_model_v6.pth \
  --output outputs/
```

This produces:
- `ensemble_0000096_overlay.png` — Segmentation overlay on input image
- `ensemble_0000096_comparison.png` — Side-by-side comparison with GT
- Console output with **mIoU**, **per-class IoU**, and **pixel accuracy**

### 4. Run Full Test Set Evaluation

```bash
python perception_engine/training/eval_batch.py \
  --weights best_model_v6.pth
```

### 5. Run the Interactive UI

```bash
# Terminal 1: Backend API
PERCEPTION_WEIGHTS=best_model_v6.pth \
PERCEPTION_PRIOR=joint_histograms.pkl \
python -m uvicorn perception_engine.api.server:app --port 8000

# Terminal 2: Frontend
cd ui && npm install && npm run dev
```

Then open `http://localhost:3000`:
- **Perception Lab** → Upload image, run ensemble pipeline, view overlay + cost map + path
- **Technical** → Full evaluation: version progression, per-class IoU, robustness, confusion matrix

---

## 🔬 Training

### Training on Kaggle (Dual T4 GPUs)

```bash
# Upload train_dinov2_v6_kaggle.py to Kaggle
# Set GPU accelerator to T4 × 2
python train_dinov2_v6_kaggle.py
```

**Training Configuration:**
- Loss: Cross-Entropy with inverse-frequency class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: Cosine Annealing with warm restarts
- Duration: 50 epochs
- Backbone: Frozen (only 2.1M-param head is trained)

### Building the 4D Prior

```bash
python perception_engine/training/build_joint_histograms.py \
  --train-dir <path_to_training_data>
```

This extracts `P(class | H, S, V, Y)` from the training images and saves it as `joint_histograms.pkl`.

---

## 🧪 Evaluation

### Metrics Used
- **Mean IoU (mIoU)**: Primary metric — average per-class Intersection over Union
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Per-Class IoU**: Individual class performance
- **Confusion Matrix**: Pixel-level misclassification patterns

### Top Confusion Pairs

| Ground Truth → Prediction | Misclassified | % of Total | Root Cause |
|---|---|---|---|
| dry_grass → rock | 32.7M pixels | 6.29% | Identical HSV color profiles |
| landscape → rock | 32.0M pixels | 6.16% | Color overlap + adjacent positions |
| rock → landscape | 16.1M pixels | 3.10% | Reverse confusion at boundaries |

### Edge Cases & Limitations
- **Rock vs Landscape**: Share identical color — mitigated by spatial prior but breaks down in flat boulder fields
- **Absent Classes**: `ground_clutter`, `flower`, `log` had zero training samples → zero predictions
- **Sparse Classes**: `lush_bush` (15K pixels / 519M total) → insufficient data for learning

---

## 🛡️ Generalization Strategy

Three layers ensure performance on unseen desert environments:

1. **Domain-Invariant Features**: Frozen DINOv2 backbone learns structural features, not texture (17% → 35% mIoU)
2. **4D Bayesian Prior**: Statistical color-position constraints from training data (sky=top, rocks=elevated) transfer across environments
3. **Class Frequency Recalibration**: Boosts underrepresented obstacle classes (rock: 3.5×) to prevent dominant-class suppression

---

## 🖥️ Interactive UI

The Next.js frontend provides:

| Page | Function |
|---|---|
| **Command Center** | System overview dashboard |
| **Perception Lab** | Upload image → run ensemble inference → view overlay, cost map, A* path |
| **Technical** | Full evaluation: version progression, per-class IoU, confusion pairs, robustness |

The Perception Lab runs the **exact same** `infer_ensemble.py` CLI as a subprocess — the UI output is byte-for-byte identical to CLI output.

---

## 📄 License

This project was developed for the SPIT Hackathon.

---

## 👥 Team

Developed as part of the perception challenge for autonomous UGV terrain analysis.
