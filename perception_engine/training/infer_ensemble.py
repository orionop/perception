#!/usr/bin/env python3
"""
FULL STACKING PIPELINE: V3 + V5 + V6 Ensemble + TTA + Post-Processing

Layers:
  1. Load 3 models (V3: DeepLabV3+, V5: DINOv2-4block, V6: DINOv2-6block)
  2. TTA: run each model on original + hflip + brightness variants
  3. Ensemble: weighted average of all probability maps
  4. Post-processing: spatial priors + impossible class remapping + CRF-lite
  5. Compute mIoU against ground truth

Usage:
    python perception_engine/training/infer_ensemble.py \
        --image path/to/color.png \
        --gt path/to/segmentation.png \
        --output outputs
"""

import argparse, os, sys, gc
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# CONFIG
# ============================================================
CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# Ensemble weights — V6 gets most weight (best domain robustness)
ENSEMBLE_WEIGHTS = {
    "v3": 0.25,
    "v5": 0.25,
    "v6": 0.50,
}

# Color palette (BGR)
PALETTE = [
    (0, 100, 255),   (0, 165, 255),  (0, 200, 0),     (0, 0, 200),
    (180, 0, 180),   (128, 128, 200),(200, 180, 220),  (0, 255, 255),
    (180, 230, 200), (255, 230, 200),
]


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

# --- DINOv2 Segmentation Head (V5: 2 blocks, V6: 3 blocks) ---
class SegHeadV5(nn.Module):
    """V5 head — 2 ConvNeXt blocks."""
    def __init__(self, in_ch, out_ch, tW, tH):
        super().__init__()
        self.H, self.W = tH, tW
        h = 256
        self.stem = nn.Sequential(nn.Conv2d(in_ch, h, 7, padding=3), nn.BatchNorm2d(h), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(h, h, 7, padding=3, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(h, h, 5, padding=2, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(h, out_ch, 1))
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x); x = x + self.block1(x); x = x + self.block2(x)
        return self.classifier(x)


class SegHeadV6(nn.Module):
    """V6 head — 3 ConvNeXt blocks."""
    def __init__(self, in_ch, out_ch, tW, tH):
        super().__init__()
        self.H, self.W = tH, tW
        h = 256
        self.stem = nn.Sequential(nn.Conv2d(in_ch, h, 7, padding=3), nn.BatchNorm2d(h), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(h, h, 7, padding=3, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(h, h, 5, padding=2, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.block3 = nn.Sequential(nn.Conv2d(h, h, 3, padding=1, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout2d(0.15), nn.Conv2d(h, out_ch, 1))
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x); x = x + self.block1(x); x = x + self.block2(x); x = x + self.block3(x)
        return self.classifier(x)


# ============================================================
# MODEL LOADERS
# ============================================================

def load_v3(weights_path, device):
    """Load V3 DeepLabV3+ ResNet50 via segmentation_models_pytorch."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("  ⚠ SMP not installed, skipping V3")
        return None, None

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    def predict_fn(img_rgb, target_h, target_w):
        """V3 inference: resize to 288x512, normalize, forward."""
        resized = cv2.resize(img_rgb, (512, 288))
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(tensor)
            # Resize logits to common target size
            logits = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu()

    return model, predict_fn


def load_dinov2(weights_path, device, version="v6"):
    """Load DINOv2 backbone + seg head (V5 or V6)."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    backbone_name = ckpt.get("backbone_name", "dinov2_vitb14_reg")
    embed_dim = ckpt.get("embed_dim", 768)
    token_w = ckpt.get("token_w", 34)
    token_h = ckpt.get("token_h", 19)
    img_size = ckpt.get("img_size", [266, 476])
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    img_h, img_w = img_size

    # Load backbone
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.to(device)

    # Restore fine-tuned blocks
    if "backbone_blocks" in ckpt:
        for key, state in ckpt["backbone_blocks"].items():
            block_idx = int(key.split("_")[1])
            backbone.blocks[block_idx].load_state_dict(state)
    if "backbone_norm" in ckpt and hasattr(backbone, 'norm'):
        backbone.norm.load_state_dict(ckpt["backbone_norm"])
    backbone.eval()

    # Load seg head
    HeadClass = SegHeadV6 if version == "v6" else SegHeadV5
    seg_head = HeadClass(embed_dim, num_classes, token_w, token_h).to(device)
    seg_head.load_state_dict(ckpt["seg_head"])
    seg_head.eval()

    def predict_fn(img_rgb, target_h, target_w):
        """DINOv2 inference."""
        resized = cv2.resize(img_rgb, (img_w, img_h))
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            features = backbone.forward_features(tensor)["x_norm_patchtokens"]
            logits = seg_head(features)
            logits = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu()

    return (backbone, seg_head), predict_fn


# ============================================================
# TTA (Test-Time Augmentation)
# ============================================================

def tta_predict(predict_fn, img_rgb, target_h, target_w):
    """Run TTA: original + hflip + brightness variants, average probs."""
    all_probs = []

    # 1. Original
    probs = predict_fn(img_rgb, target_h, target_w)
    all_probs.append(probs)

    # 2. Horizontal flip
    flipped = img_rgb[:, ::-1, :].copy()
    probs_flip = predict_fn(flipped, target_h, target_w)
    # Flip back
    probs_flip = torch.flip(probs_flip, dims=[3])
    all_probs.append(probs_flip)

    # 3. Brightness +15%
    bright = np.clip(img_rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    probs_bright = predict_fn(bright, target_h, target_w)
    all_probs.append(probs_bright)

    # 4. Brightness -15%
    dark = np.clip(img_rgb.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    probs_dark = predict_fn(dark, target_h, target_w)
    all_probs.append(probs_dark)

    # Average
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs


# ============================================================
# POST-PROCESSING (EDA-derived rules + Rock Rescue + CRF)
# ============================================================

def rock_rescue(corrected, image_rgb, confidence_map, probs):
    """Layer 1: Color-based rock rescue using HSV thresholds from training EDA.

    From training EDA:
        Rock:      H=7-12,  S=97-150, V=28-189 (avg S=121, V=123)
        Landscape: H=10-14, S=81-121, V=66-178 (avg S=100, V=135)
        Dry_grass: H=9-13,  S=99-166, V=39-146 (avg S=133, V=91) ← overlaps!

    Key separator: rocks are MORE SATURATED and DARKER than landscape.
    BUT dry_grass also has high saturation → must be careful.
    Only target landscape(8) predictions, NOT dry_grass(2).
    """
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H = image_hsv[:, :, 0].astype(np.float32)
    S = image_hsv[:, :, 1].astype(np.float32)
    V = image_hsv[:, :, 2].astype(np.float32)

    corrections = 0
    h_img = corrected.shape[0]

    # Rock color: warm hue, high saturation, DARKER
    rock_color = (
        (H >= 5) & (H <= 14) &      # warm hue (same as landscape)
        (S >= 115) &                  # saturation ABOVE landscape avg (100)
        (V <= 145) &                  # brightness BELOW landscape avg (135)
        (V >= 25)                     # not pure black
    )

    # Primary rule: landscape(8) predictions that color-match rock
    # Only in bottom 75% of image (Y > 0.25) — rocks are on the ground
    spatial_ok = np.zeros_like(corrected, dtype=bool)
    spatial_start = int(h_img * 0.25)
    spatial_ok[spatial_start:, :] = True

    candidates_landscape = (
        (corrected == 8) &            # predicted as landscape
        rock_color &                   # but color matches rock
        spatial_ok &                   # in ground region
        (confidence_map < 0.55)        # model not confident
    )
    corrected[candidates_landscape] = 7  # rock
    corrections += int(candidates_landscape.sum())

    # Secondary rule: dry_bush(3) in bottom half that are very dark → shadow rocks
    mid_start = int(h_img * 0.4)
    shadow_rock = (
        (corrected[mid_start:, :] == 3) &       # dry_bush
        (V[mid_start:, :] < 70) &                # very dark
        (S[mid_start:, :] > 90) &                # saturated
        (confidence_map[mid_start:, :] < 0.45)   # low confidence
    )
    corrected[mid_start:, :][shadow_rock] = 7
    corrections += int(shadow_rock.sum())

    return corrected, corrections


def bilateral_crf(corrected, image_rgb, probs, iterations=5):
    """Layer 2: Bilateral-filter CRF approximation.

    Uses the original image color to refine mask boundaries.
    Principle: nearby pixels with similar color should have the same class.

    This is a lightweight alternative to pydensecrf that uses OpenCV's
    bilateral filter on the probability maps, weighted by image color similarity.
    """
    h, w = corrected.shape
    refined_probs = probs.copy()  # (C, H, W)

    # Bilateral filter each class probability channel
    # The filter smooths probabilities but preserves edges where image color changes
    for iteration in range(iterations):
        for c in range(probs.shape[0]):
            prob_channel = (refined_probs[c] * 255).astype(np.uint8)
            # Joint bilateral: smooth probability using image color as guide
            # d=9: neighborhood size, sigmaColor=75: color similarity, sigmaSpace=13: spatial
            filtered = cv2.bilateralFilter(prob_channel, d=9, sigmaColor=75, sigmaSpace=13)
            refined_probs[c] = filtered.astype(np.float32) / 255.0

        # Re-normalize probabilities
        prob_sum = refined_probs.sum(axis=0, keepdims=True)
        prob_sum[prob_sum == 0] = 1.0
        refined_probs = refined_probs / prob_sum

    # New prediction from refined probabilities
    new_mask = refined_probs.argmax(axis=0).astype(np.uint8)
    changes = (new_mask != corrected).sum()
    return new_mask, int(changes)


def postprocess_mask(pred_mask, confidence_map, probs, orig_h, orig_w, image_rgb=None):
    """Apply all post-processing layers.

    Layer 0: Remap impossible classes + spatial priors (basic rules)
    Layer 1: Rock rescue via HSV color thresholds
    Layer 2: Bilateral CRF boundary refinement
    """
    corrected = pred_mask.copy()
    total_corrections = 0

    # ============================
    # LAYER 0: Basic rules
    # ============================
    print("    Layer 0: Basic rules (impossible classes, spatial priors)...")

    # Rule 0a: Remap impossible test classes (vectorized)
    impossible_classes = [4, 5, 6]  # ground_clutter, flower, log
    impossible_mask = np.isin(corrected, impossible_classes)
    if impossible_mask.sum() > 0:
        probs_valid = probs.copy()
        for ic in impossible_classes:
            probs_valid[ic, :, :] = 0
        corrected[impossible_mask] = probs_valid.argmax(axis=0)[impossible_mask]
        n = int(impossible_mask.sum())
        total_corrections += n
        print(f"      Remapped {n:,} impossible class pixels")

    # Rule 0b: Sky spatial prior — top 10%, low confidence → sky
    sky_region_h = int(orig_h * 0.10)
    sky_override = (corrected[:sky_region_h, :] != 9) & (confidence_map[:sky_region_h, :] < 0.5)
    corrected[:sky_region_h, :][sky_override] = 9
    n = int(sky_override.sum())
    total_corrections += n
    if n > 0:
        print(f"      Sky override: {n:,} pixels")

    # Rule 0c: Horizon — below 85%, no sky
    bottom_cutoff = int(orig_h * 0.85)
    sky_in_bottom = corrected[bottom_cutoff:, :] == 9
    if sky_in_bottom.sum() > 0:
        probs_bottom = probs[:, bottom_cutoff:, :].copy()
        probs_bottom[9, :, :] = 0
        corrected[bottom_cutoff:, :][sky_in_bottom] = probs_bottom.argmax(axis=0)[sky_in_bottom]
        n = int(sky_in_bottom.sum())
        total_corrections += n
        print(f"      Horizon fix: {n:,} pixels")

    # Rule 0d: Tree spatial — below 60% + low conf → not tree
    tree_cutoff = int(orig_h * 0.60)
    tree_below = (corrected[tree_cutoff:, :] == 0) & (confidence_map[tree_cutoff:, :] < 0.5)
    if tree_below.sum() > 0:
        probs_low = probs[:, tree_cutoff:, :].copy()
        probs_low[0, :, :] = 0
        corrected[tree_cutoff:, :][tree_below] = probs_low.argmax(axis=0)[tree_below]
        n = int(tree_below.sum())
        total_corrections += n
        print(f"      Tree fix: {n:,} pixels")

    # ============================
    # LAYER 1: Rock rescue
    # ============================
    if image_rgb is not None:
        print("    Layer 1: Rock rescue (HSV color thresholds)...")
        corrected, rock_corrections = rock_rescue(corrected, image_rgb, confidence_map, probs)
        total_corrections += rock_corrections
        print(f"      Rescued {rock_corrections:,} pixels as rock")

    # ============================
    # LAYER 2: Bilateral CRF
    # ============================
    if image_rgb is not None:
        print("    Layer 2: Bilateral CRF (boundary refinement)...")
        # Build refined probs from current corrected mask (one-hot) blended with model probs
        # Use 70% corrected + 30% original probs for stability
        corrected_onehot = np.eye(NUM_CLASSES)[corrected].transpose(2, 0, 1).astype(np.float32)
        blended_probs = 0.7 * corrected_onehot + 0.3 * probs
        corrected, crf_changes = bilateral_crf(corrected, image_rgb, blended_probs, iterations=3)
        total_corrections += crf_changes
        print(f"      CRF refined {crf_changes:,} pixels")

    # ============================
    # LAYER 3: Small region cleanup (after everything)
    # ============================
    print("    Layer 3: Small region cleanup...")
    small_fixes = 0
    for cls_id in range(NUM_CLASSES):
        cls_mask = (corrected == cls_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 50:
                region = labels == label_id
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(region.astype(np.uint8), kernel)
                neighbors = dilated.astype(bool) & ~region
                if neighbors.sum() > 0:
                    neighbor_classes = corrected[neighbors]
                    majority = np.bincount(neighbor_classes, minlength=NUM_CLASSES).argmax()
                    corrected[region] = majority
                    small_fixes += int(region.sum())
    total_corrections += small_fixes
    print(f"      Cleaned {small_fixes:,} small region pixels")

    return corrected, total_corrections


# ============================================================
# METRICS
# ============================================================

def remap_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask


def compute_metrics(pred, gt, num_classes=NUM_CLASSES):
    ious = []
    for cls in range(num_classes):
        pred_c = pred == cls
        gt_c = gt == cls
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            ious.append(float(inter / union))
        else:
            ious.append(float('nan'))
    valid = [v for v in ious if not np.isnan(v)]
    miou = np.mean(valid) if valid else 0.0
    pixel_acc = (pred == gt).sum() / pred.size
    return miou, pixel_acc, ious


def create_overlay(image, mask, alpha=0.45):
    overlay = image.copy()
    for cls_id in range(NUM_CLASSES):
        overlay[mask == cls_id] = PALETTE[cls_id]
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ensemble V3+V5+V6 + TTA + Post-Processing")
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--v3-weights", default="best_model_v3.pth")
    parser.add_argument("--v5-weights", default="best_model_v5.pth")
    parser.add_argument("--v6-weights", default="best_model_v6.pth")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--no-tta", action="store_true", help="Skip TTA")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip post-processing")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load image
    image_bgr = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_bgr.shape[:2]
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    # Common target size for probability averaging
    target_h, target_w = orig_h, orig_w

    # ---- Load models ----
    models = {}
    predict_fns = {}

    print("\n=== Loading Models ===")

    # V3
    if os.path.exists(args.v3_weights):
        print(f"  Loading V3 (DeepLabV3+ ResNet50)...")
        try:
            model, fn = load_v3(args.v3_weights, device)
            if fn is not None:
                models["v3"] = model
                predict_fns["v3"] = fn
                print(f"  ✅ V3 loaded")
        except Exception as e:
            print(f"  ⚠ V3 failed: {e}")
    else:
        print(f"  ⚠ V3 weights not found: {args.v3_weights}")

    # V5
    if os.path.exists(args.v5_weights):
        print(f"  Loading V5 (DINOv2 4-block)...")
        try:
            model, fn = load_dinov2(args.v5_weights, device, version="v5")
            models["v5"] = model
            predict_fns["v5"] = fn
            print(f"  ✅ V5 loaded")
        except Exception as e:
            print(f"  ⚠ V5 failed: {e}")
    else:
        print(f"  ⚠ V5 weights not found: {args.v5_weights}")

    # V6
    if os.path.exists(args.v6_weights):
        print(f"  Loading V6 (DINOv2 6-block domain-robust)...")
        try:
            model, fn = load_dinov2(args.v6_weights, device, version="v6")
            models["v6"] = model
            predict_fns["v6"] = fn
            print(f"  ✅ V6 loaded")
        except Exception as e:
            print(f"  ⚠ V6 failed: {e}")
    else:
        print(f"  ⚠ V6 weights not found: {args.v6_weights}")

    if not predict_fns:
        print("ERROR: No models loaded!")
        sys.exit(1)

    print(f"\n  Active models: {list(predict_fns.keys())}")

    # Normalize weights for available models
    active_weights = {k: ENSEMBLE_WEIGHTS[k] for k in predict_fns}
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    print(f"  Ensemble weights: {active_weights}")

    # ---- Run inference ----
    print("\n=== Running Inference ===")
    ensemble_probs = torch.zeros(1, NUM_CLASSES, target_h, target_w)

    for name, fn in predict_fns.items():
        print(f"  {name}...", end=" ", flush=True)
        if args.no_tta:
            probs = fn(image_rgb, target_h, target_w)
        else:
            probs = tta_predict(fn, image_rgb, target_h, target_w)

        ensemble_probs += probs * active_weights[name]
        print(f"done (weight={active_weights[name]:.2f})")

    # ---- Apply EDA P(class|H,S,V,Y) Prior ----
    print("\n=== Bayesian Prior Injection ===")
    probs_injected = ensemble_probs.squeeze().numpy()  # (C, H, W)

    joint_path = os.path.join(os.path.dirname(args.image), "..", "joint_histograms.pkl")
    if not os.path.exists(joint_path):
        joint_path = "joint_histograms.pkl"

    if os.path.exists(joint_path):
        import pickle
        with open(joint_path, 'rb') as f:
            joint_data = pickle.load(f)

        # Re-use logic from eval_batch.py
        hists = joint_data['histograms']
        h_bins = joint_data['h_bins']
        s_bins = joint_data['s_bins']
        v_bins = joint_data['v_bins']
        y_bins = joint_data['y_bins']

        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        H = image_hsv[:, :, 0].astype(np.float32)
        S = image_hsv[:, :, 1].astype(np.float32)
        V = image_hsv[:, :, 2].astype(np.float32)
        Y = np.arange(orig_h, dtype=np.float32)[:, None] * np.ones(orig_w)[None, :]

        h_idx = np.clip((H / 180 * h_bins).astype(int), 0, h_bins - 1)
        s_idx = np.clip((S / 256 * s_bins).astype(int), 0, s_bins - 1)
        v_idx = np.clip((V / 256 * v_bins).astype(int), 0, v_bins - 1)
        y_idx = np.clip((Y / orig_h * y_bins).astype(int), 0, y_bins - 1)

        flat_idx = (h_idx * (s_bins * v_bins * y_bins) +
                    s_idx * (v_bins * y_bins) +
                    v_idx * y_bins +
                    y_idx).astype(int).flatten()

        joint_probs = np.zeros((NUM_CLASSES, orig_h, orig_w), dtype=np.float32)
        for cls in range(NUM_CLASSES):
            joint_probs[cls] = hists[cls][flat_idx].reshape(orig_h, orig_w)

        for c in [1, 4, 5, 6]:
            joint_probs[c, :, :] = 0

        jp_sum = joint_probs.sum(axis=0, keepdims=True)
        jp_sum[jp_sum == 0] = 1.0
        joint_probs = joint_probs / jp_sum

        freq_weight = np.ones((NUM_CLASSES, 1, 1), dtype=np.float32)
        freq_weight[7, 0, 0] = 3.5   # rock
        freq_weight[2, 0, 0] = 0.60  # dry_grass
        freq_weight[3, 0, 0] = 0.65  # dry_bush
        freq_weight[0, 0, 0] = 0.40  # tree

        eps = 1e-8
        dnn_signal = np.maximum(probs_injected, eps)
        joint_signal = np.maximum(joint_probs, eps)
        
        combined = (
            np.power(dnn_signal, 0.40) *
            np.power(joint_signal, 0.55) *
            freq_weight
        )

        prob_sum = combined.sum(axis=0, keepdims=True)
        prob_sum[prob_sum == 0] = 1.0
        probs_injected = combined / prob_sum
        print("  Applied 4D Joint Histogram Fusion (0.53 mIoU config)")
    else:
        print("  ⚠ No joint_histograms.pkl found, using DNN only")

    # Final prediction
    pred_mask = probs_injected.argmax(axis=0).astype(np.uint8)
    confidence = probs_injected.max(axis=0)
    probs_np = probs_injected

    print(f"  Final prediction: {pred_mask.shape}")

    # ---- Post-processing ----
    if not args.no_postprocess:
        print("\n=== Post-Processing ===")
        pred_final, n_corrections = postprocess_mask(pred_mask, confidence, probs_np, orig_h, orig_w, image_rgb=image_rgb)
        pct = n_corrections / pred_mask.size * 100
        print(f"  Corrected {n_corrections:,} pixels ({pct:.1f}%)")
    else:
        pred_final = pred_mask
        print("  Post-processing skipped")

    # ---- Save outputs ----
    basename = os.path.splitext(os.path.basename(args.image))[0]

    # Overlay
    overlay = create_overlay(image_bgr, pred_final)
    overlay_path = os.path.join(args.output, f"ensemble_{basename}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"\n  Saved: {overlay_path}")

    # Save raw class-index mask for API cost mapping / path planning
    mask_path = os.path.join(args.output, f"ensemble_{basename}_mask.npy")
    np.save(mask_path, pred_final.astype(np.uint8))

    # Also save raw ensemble (no post-processing) overlay for comparison
    overlay_raw = create_overlay(image_bgr, pred_mask)
    cv2.imwrite(os.path.join(args.output, f"ensemble_{basename}_raw_overlay.png"), overlay_raw)

    # ---- Evaluate ----
    if args.gt and os.path.exists(args.gt):
        gt_raw = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
        if len(gt_raw.shape) == 3:
            gt_raw = gt_raw[:, :, 0]
        gt_remapped = remap_mask(gt_raw)
        gt_resized = cv2.resize(gt_remapped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Metrics WITHOUT post-processing
        miou_raw, acc_raw, per_class_raw = compute_metrics(pred_mask, gt_resized)

        # Metrics WITH post-processing
        miou_pp, acc_pp, per_class_pp = compute_metrics(pred_final, gt_resized)

        print(f"\n{'='*65}")
        print(f"  {'':>20} {'RAW ENSEMBLE':>14} {'+ POST-PROC':>14}")
        print(f"  {'mIoU:':>20} {miou_raw:>14.4f} {miou_pp:>14.4f}")
        print(f"  {'Pixel Acc:':>20} {acc_raw:>14.4f} {acc_pp:>14.4f}")
        print(f"  {'Improvement:':>20} {'':>14} {'+' if miou_pp > miou_raw else ''}{(miou_pp - miou_raw)*100:>13.2f}%")
        print(f"{'='*65}")

        print(f"\n  {'Class':>16} | {'Raw':>8} | {'+ PP':>8} | {'Gain':>8}")
        print(f"  {'-'*50}")
        for i, name in enumerate(CLASS_NAMES):
            r = per_class_raw[i]
            p = per_class_pp[i]
            r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
            p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
            gain = ""
            if not np.isnan(r) and not np.isnan(p):
                diff = p - r
                gain = f"{diff:+.3f}" if abs(diff) > 0.001 else "="
            print(f"  {name:>16} | {r_str:>8} | {p_str:>8} | {gain:>8}")

        # Comparison figure
        gt_overlay = create_overlay(image_bgr, gt_resized)
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(
            f"Ensemble (V3+V5+V6) + TTA + PP — mIoU: {miou_raw:.4f} → {miou_pp:.4f}",
            fontsize=15, fontweight='bold'
        )
        titles = ["Original", "Raw Ensemble", "+ Post-Processing", "Ground Truth"]
        images = [
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay_raw, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB),
        ]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img); ax.set_title(title, fontsize=12); ax.axis('off')

        patches = []
        for i, name in enumerate(CLASS_NAMES):
            c = [PALETTE[i][2]/255, PALETTE[i][1]/255, PALETTE[i][0]/255]
            p_str = f"{per_class_pp[i]:.3f}" if not np.isnan(per_class_pp[i]) else "N/A"
            patches.append(mpatches.Patch(color=c, label=f"{name}: {p_str}"))
        fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.08))

        comp_path = os.path.join(args.output, f"ensemble_{basename}_comparison.png")
        plt.tight_layout()
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {comp_path}")
    else:
        print("\n  No ground truth — skipping metrics")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
