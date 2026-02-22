#!/usr/bin/env python3
"""Standalone DINOv2 V5 inference + visualization script.

Usage:
    python perception_engine/training/infer_dinov2.py \
        --image path/to/color.png \
        --gt path/to/segmentation.png \
        --weights best_model_v5.pth
"""
import argparse, os, yaml, sys
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


# ---- Segmentation Head (must match training) ----
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.15),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.classifier(x)


# ---- Class definitions ----
CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10

# Color palette (BGR for overlay, matching perception engine)
PALETTE = [
    (0, 100, 255),   # tree - blue
    (0, 165, 255),   # lush_bush - orange
    (0, 200, 0),     # dry_grass - green
    (0, 0, 200),     # dry_bush - red
    (180, 0, 180),   # ground_clutter - purple
    (128, 128, 200), # flower - salmon
    (200, 180, 220), # log - pink
    (180, 180, 180), # rock - gray
    (180, 230, 200), # landscape - light yellow
    (255, 230, 200), # sky - light blue
]

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


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

    correct = (pred == gt).sum()
    total = pred.size
    pixel_acc = correct / total

    return miou, pixel_acc, ious


def create_overlay(image, mask, alpha=0.45):
    """Blend colored mask on image."""
    overlay = image.copy()
    for cls_id in range(NUM_CLASSES):
        color = PALETTE[cls_id]
        overlay[mask == cls_id] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended


def create_comparison_figure(image_rgb, pred_overlay_rgb, gt_overlay_rgb,
                             pred_mask, gt_mask, miou, pixel_acc, per_class, save_path):
    """Create a nice comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"DINOv2 V5 — mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f}",
                 fontsize=16, fontweight='bold')

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(pred_overlay_rgb)
    axes[1].set_title("V5 Prediction", fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(gt_overlay_rgb)
    axes[2].set_title("Ground Truth", fontsize=13)
    axes[2].axis('off')

    # Legend
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        c = [PALETTE[i][2]/255, PALETTE[i][1]/255, PALETTE[i][0]/255]  # BGR→RGB
        iou_str = f"{per_class[i]:.3f}" if not np.isnan(per_class[i]) else "N/A"
        patches.append(mpatches.Patch(color=c, label=f"{name}: {iou_str}"))
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Color image path")
    parser.add_argument("--gt", default=None, help="Ground truth mask path")
    parser.add_argument("--weights", default="best_model_v5.pth")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"Loading V5 weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)

    backbone_name = ckpt.get("backbone_name", "dinov2_vitb14_reg")
    embed_dim = ckpt.get("embed_dim", 768)
    token_w = ckpt.get("token_w", 34)
    token_h = ckpt.get("token_h", 19)
    img_size = ckpt.get("img_size", [266, 476])
    num_classes = ckpt.get("num_classes", NUM_CLASSES)

    img_h, img_w = img_size

    # ---- Load DINOv2 backbone ----
    print(f"Loading {backbone_name}...")
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.to(device)

    # Restore fine-tuned blocks if available
    if "backbone_blocks" in ckpt:
        print(f"Restoring {len(ckpt['backbone_blocks'])} fine-tuned blocks...")
        for key, state in ckpt["backbone_blocks"].items():
            block_idx = int(key.split("_")[1])
            backbone.blocks[block_idx].load_state_dict(state)
    if "backbone_norm" in ckpt and hasattr(backbone, 'norm'):
        backbone.norm.load_state_dict(ckpt["backbone_norm"])

    backbone.eval()

    # ---- Load seg head ----
    seg_head = SegmentationHead(embed_dim, num_classes, token_w, token_h).to(device)
    seg_head.load_state_dict(ckpt["seg_head"])
    seg_head.eval()

    print(f"Model loaded! Input size: {img_h}x{img_w}, tokens: {token_h}x{token_w}")

    # ---- Load and preprocess image ----
    image_bgr = cv2.imread(args.image)
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input
    resized = cv2.resize(image_rgb, (img_w, img_h))
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # ---- Inference ----
    print("Running inference...")
    with torch.no_grad():
        features = backbone.forward_features(tensor)["x_norm_patchtokens"]
        logits = seg_head(features)
        logits = F.interpolate(logits, size=(img_h, img_w), mode="bilinear", align_corners=False)
        pred_mask = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    # Confidence map
    probs = torch.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values.squeeze().cpu().numpy()
    conf_mean = confidence.mean()

    # Scale prediction to original resolution
    pred_full = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # ---- Create overlays ----
    pred_overlay = create_overlay(image_bgr, pred_full)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    # Save prediction overlay
    overlay_path = os.path.join(args.output, f"dinov2_v5_{basename}_overlay.png")
    cv2.imwrite(overlay_path, pred_overlay)
    print(f"  Saved: {overlay_path}")

    # Save confidence map
    conf_resized = cv2.resize(confidence, (orig_w, orig_h))
    conf_colored = cv2.applyColorMap((conf_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    conf_path = os.path.join(args.output, f"dinov2_v5_{basename}_confidence.png")
    cv2.imwrite(conf_path, conf_colored)
    print(f"  Saved: {conf_path}")

    # ---- Ground truth comparison ----
    if args.gt and os.path.exists(args.gt):
        gt_raw = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
        if len(gt_raw.shape) == 3:
            gt_raw = gt_raw[:, :, 0]
        gt_remapped = remap_mask(gt_raw)
        gt_resized = cv2.resize(gt_remapped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        miou, pixel_acc, per_class = compute_metrics(pred_full, gt_resized)

        print(f"\n{'='*60}")
        print(f"  mIoU:       {miou:.4f}")
        print(f"  Pixel Acc:  {pixel_acc:.4f}")
        print(f"  Confidence: {conf_mean:.4f}")
        print(f"{'='*60}")
        print(f"  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = per_class[i]
            val_str = f"{v:.3f}" if not np.isnan(v) else "N/A  "
            bar = "█" * int(v * 20) if not np.isnan(v) else ""
            print(f"    {name:>16}: {val_str} {bar}")

        # Create GT overlay and comparison figure
        gt_overlay = create_overlay(image_bgr, gt_resized)
        gt_overlay_path = os.path.join(args.output, f"dinov2_v5_{basename}_gt_overlay.png")
        cv2.imwrite(gt_overlay_path, gt_overlay)

        comparison_path = os.path.join(args.output, f"dinov2_v5_{basename}_comparison.png")
        create_comparison_figure(
            image_rgb,
            cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB),
            pred_full, gt_resized, miou, pixel_acc, per_class,
            comparison_path,
        )
    else:
        print(f"\n  Confidence: {conf_mean:.4f}")
        print(f"  No ground truth provided, skipping metrics.")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
