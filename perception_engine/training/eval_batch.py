#!/usr/bin/env python3
"""
Fast batch evaluation on the full test set.
Tests V6 alone (fast) + color priors + post-processing on ALL images.
Reports per-class IoU and aggregate mIoU.
"""

import os, sys, gc, time, pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================
# V6 Model
# ============================================================
class SegHeadV6(nn.Module):
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


def remap_mask(mask):
    new_mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask


def apply_color_priors(probs, image_rgb, orig_h, orig_w, hist_data=None, eda_data=None, joint_data=None):
    """Apply comprehensive EDA-derived Bayesian priors.
    
    Supports 4D (HSV+Y) or 5D (HSV+Y+texture) joint histograms.
    Also applies superpixel voting and connected component cleaning.
    """
    # ---- 1. Suppress impossible/hallucinated classes ----
    probs[4, :, :] *= 0.0
    probs[5, :, :] *= 0.0
    probs[6, :, :] *= 0.0
    probs[1, :, :] *= 0.01

    # ---- 2. Joint histogram lookup ----
    joint_probs = None
    if joint_data is not None:
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hists = joint_data['histograms']
        h_bins = joint_data['h_bins']
        s_bins = joint_data['s_bins']
        v_bins = joint_data['v_bins']
        y_bins = joint_data['y_bins']

        H = image_hsv[:, :, 0].astype(np.float32)
        S = image_hsv[:, :, 1].astype(np.float32)
        V = image_hsv[:, :, 2].astype(np.float32)
        Y = np.arange(orig_h, dtype=np.float32)[:, None] * np.ones(orig_w)[None, :]

        h_idx = np.clip((H / 180 * h_bins).astype(int), 0, h_bins - 1)
        s_idx = np.clip((S / 256 * s_bins).astype(int), 0, s_bins - 1)
        v_idx = np.clip((V / 256 * v_bins).astype(int), 0, v_bins - 1)
        y_idx = np.clip((Y / orig_h * y_bins).astype(int), 0, y_bins - 1)

        is_5d = joint_data.get('type') == 'joint_hsv_y_texture'
        if is_5d:
            t_bins = joint_data['t_bins']
            t_max = joint_data['t_max']
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            local_mean = cv2.blur(gray, (7, 7))
            local_sq = cv2.blur(gray ** 2, (7, 7))
            local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
            t_idx = np.clip((local_std / t_max * t_bins).astype(int), 0, t_bins - 1)

            flat_idx = (h_idx * (s_bins * v_bins * y_bins * t_bins) +
                        s_idx * (v_bins * y_bins * t_bins) +
                        v_idx * (y_bins * t_bins) +
                        y_idx * t_bins +
                        t_idx).astype(int).flatten()
        else:
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

    # Fallback: separate color histogram
    color_probs = None
    if joint_probs is None and hist_data is not None:
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hists = hist_data['histograms']
        h_bins, s_bins, v_bins = hist_data['h_bins'], hist_data['s_bins'], hist_data['v_bins']
        H = image_hsv[:, :, 0].astype(np.float32)
        S = image_hsv[:, :, 1].astype(np.float32)
        V = image_hsv[:, :, 2].astype(np.float32)
        h_idx = np.clip((H / 180 * h_bins).astype(int), 0, h_bins - 1)
        s_idx = np.clip((S / 256 * s_bins).astype(int), 0, s_bins - 1)
        v_idx = np.clip((V / 256 * v_bins).astype(int), 0, v_bins - 1)
        flat_idx = (h_idx * (s_bins * v_bins) + s_idx * v_bins + v_idx).astype(int).flatten()
        color_probs = np.zeros((NUM_CLASSES, orig_h, orig_w), dtype=np.float32)
        for cls in range(NUM_CLASSES):
            color_probs[cls] = hists[cls][flat_idx].reshape(orig_h, orig_w)
        for c in [1, 4, 5, 6]:
            color_probs[c, :, :] = 0
        hp_sum = color_probs.sum(axis=0, keepdims=True)
        hp_sum[hp_sum == 0] = 1.0
        color_probs = color_probs / hp_sum

    # ---- 3. Frequency recalibration ----
    freq_weight = np.ones((NUM_CLASSES, 1, 1), dtype=np.float32)
    if eda_data is not None:
        freq_weight[7, 0, 0] = 3.5   # rock
        freq_weight[2, 0, 0] = 0.60  # dry_grass
        freq_weight[3, 0, 0] = 0.65  # dry_bush
        freq_weight[0, 0, 0] = 0.40  # tree

    # ---- 4. Multiplicative Bayesian fusion ----
    eps = 1e-8
    dnn_signal = np.maximum(probs, eps)

    if joint_probs is not None:
        joint_signal = np.maximum(joint_probs, eps)
        combined = (
            np.power(dnn_signal, 0.40) *
            np.power(joint_signal, 0.55) *
            freq_weight
        )
    elif color_probs is not None:
        color_signal = np.maximum(color_probs, eps)
        combined = (
            np.power(dnn_signal, 0.50) *
            np.power(color_signal, 0.40) *
            freq_weight
        )
    else:
        combined = dnn_signal * freq_weight

    # ---- 5. Normalize ----
    prob_sum = combined.sum(axis=0, keepdims=True)
    prob_sum[prob_sum == 0] = 1.0
    combined = combined / prob_sum

    return combined


def superpixel_voting(pred_mask, image_rgb, n_segments=400):
    """Apply SLIC superpixels + majority voting for spatial coherence."""
    # Use OpenCV SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(image_rgb, algorithm=cv2.ximgproc.SLICO,
                                              region_size=20, ruler=10.0)
    slic.iterate(10)
    labels = slic.getLabels()

    refined = pred_mask.copy()
    for seg_id in range(labels.max() + 1):
        mask = labels == seg_id
        if mask.sum() < 5:
            continue
        # Majority vote within superpixel
        segment_classes = pred_mask[mask]
        counts = np.bincount(segment_classes, minlength=NUM_CLASSES)
        majority = counts.argmax()
        refined[mask] = majority

    return refined


def clean_small_components(pred_mask, min_size=300):
    """Remove small connected components and replace with neighbor class."""
    cleaned = pred_mask.copy()
    for cls in range(NUM_CLASSES):
        binary = (pred_mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_size:
                # Replace with most common neighbor class
                component_mask = labels == label_id
                # Dilate to find neighbors
                dilated = cv2.dilate(component_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
                border = dilated.astype(bool) & ~component_mask
                if border.sum() > 0:
                    neighbor_classes = pred_mask[border]
                    replacement = np.bincount(neighbor_classes, minlength=NUM_CLASSES).argmax()
                    cleaned[component_mask] = replacement
    return cleaned


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best_model_v6.pth")
    parser.add_argument("--v3-weights", default="best_model.pth")
    parser.add_argument("--test-dir", default="perception_engine/Offroad_Segmentation_testImages")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-priors", action="store_true")
    parser.add_argument("--histograms", default="color_histograms.pkl")
    parser.add_argument("--eda-priors", default="eda_priors.pkl")
    parser.add_argument("--joint-histograms", default="joint_histograms.pkl")
    parser.add_argument("--multi-scale", action="store_true", default=True)
    parser.add_argument("--no-multi-scale", dest="multi_scale", action="store_false")
    parser.add_argument("--no-v3", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load histogram data if available
    hist_data = None
    if not args.no_priors and os.path.exists(args.histograms):
        with open(args.histograms, 'rb') as f:
            hist_data = pickle.load(f)
        print(f"Loaded color histograms from {args.histograms}")

    # Load EDA priors
    eda_data = None
    if not args.no_priors and os.path.exists(args.eda_priors):
        with open(args.eda_priors, 'rb') as f:
            eda_data = pickle.load(f)
        print(f"Loaded EDA priors from {args.eda_priors}")

    # Load joint histograms
    joint_data = None
    if not args.no_priors and os.path.exists(args.joint_histograms):
        with open(args.joint_histograms, 'rb') as f:
            joint_data = pickle.load(f)
        print(f"Loaded joint histograms from {args.joint_histograms}")

    # ---- Load V6 ----
    print("Loading V6...")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    backbone = torch.hub.load("facebookresearch/dinov2", ckpt.get("backbone_name", "dinov2_vitb14_reg"))
    backbone.to(device)
    if "backbone_blocks" in ckpt:
        for key, state in ckpt["backbone_blocks"].items():
            block_idx = int(key.split("_")[1])
            backbone.blocks[block_idx].load_state_dict(state)
    if "backbone_norm" in ckpt and hasattr(backbone, 'norm'):
        backbone.norm.load_state_dict(ckpt["backbone_norm"])
    backbone.eval()

    img_size = ckpt.get("img_size", [266, 476])
    seg_head = SegHeadV6(768, NUM_CLASSES, 34, 19).to(device)
    seg_head.load_state_dict(ckpt["seg_head"])
    seg_head.eval()

    # Multi-scale factors (DINOv2 needs sizes divisible by 14)
    if args.multi_scale:
        scales = [0.75, 1.0, 1.25]
        # Compute valid sizes for each scale (must be divisible by 14)
        scale_sizes = []
        for s in scales:
            h = int(img_size[0] * s)
            w = int(img_size[1] * s)
            h = (h // 14) * 14
            w = (w // 14) * 14
            scale_sizes.append((h, w))
        print(f"  Multi-scale sizes: {scale_sizes}")
    else:
        scale_sizes = [(img_size[0], img_size[1])]

    # ---- Load V3 (optional) ----
    v3_model = None
    v3_predict = None
    if not args.no_v3 and os.path.exists(args.v3_weights):
        try:
            import segmentation_models_pytorch as smp
            v3_model = smp.DeepLabV3Plus(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES
            )
            v3_ckpt = torch.load(args.v3_weights, map_location=device, weights_only=False)
            if isinstance(v3_ckpt, dict) and "model_state_dict" in v3_ckpt:
                v3_model.load_state_dict(v3_ckpt["model_state_dict"])
            else:
                v3_model.load_state_dict(v3_ckpt)
            v3_model.to(device).eval()
            print("  ✅ V3 loaded")
        except Exception as e:
            print(f"  ⚠ V3 failed: {e}")
            v3_model = None

    img_dir = os.path.join(args.test_dir, "Color_Images")
    gt_dir = os.path.join(args.test_dir, "Segmentation")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    if args.max_images:
        indices = np.linspace(0, len(files)-1, args.max_images, dtype=int)
        files = [files[i] for i in indices]

    print(f"Evaluating {len(files)} images...")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    start_time = time.time()
    for idx, fname in enumerate(files):
        img_bgr = cv2.imread(os.path.join(img_dir, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_bgr.shape[:2]

        gt_raw = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_UNCHANGED)
        if len(gt_raw.shape) == 3:
            gt_raw = gt_raw[:, :, 0]
        gt = remap_mask(gt_raw)

        # ---- V6 multi-scale + flip inference ----
        all_probs = []
        with torch.no_grad():
            for sh, sw in scale_sizes:
                th, tw = sh // 14, sw // 14
                # Need a temporary head for non-standard sizes
                if (th, tw) != (19, 34):
                    tmp_head = SegHeadV6(768, NUM_CLASSES, tw, th).to(device)
                    tmp_head.load_state_dict(seg_head.state_dict())
                    tmp_head.eval()
                    head = tmp_head
                else:
                    head = seg_head

                for flip in [False, True]:
                    img_in = img_rgb[:, ::-1, :].copy() if flip else img_rgb
                    resized = cv2.resize(img_in, (sw, sh))
                    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
                    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

                    features = backbone.forward_features(tensor)["x_norm_patchtokens"]
                    logits = head(features)
                    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                    p = torch.softmax(logits, dim=1).squeeze()
                    if flip:
                        p = torch.flip(p, dims=[2])
                    all_probs.append(p)

        # Average V6 multi-scale probs
        v6_probs = torch.stack(all_probs).mean(dim=0).numpy()

        # ---- V3 inference (single-scale + flip) ----
        if v3_model is not None:
            v3_plist = []
            with torch.no_grad():
                for flip in [False, True]:
                    img_in = img_rgb[:, ::-1, :].copy() if flip else img_rgb
                    resized = cv2.resize(img_in, (512, 288))
                    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
                    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    logits = v3_model(tensor)
                    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                    p = torch.softmax(logits, dim=1).squeeze()
                    if flip:
                        p = torch.flip(p, dims=[2])
                    v3_plist.append(p)
            v3_probs = torch.stack(v3_plist).mean(dim=0).numpy()
            # Ensemble: 75% V6, 25% V3
            probs = 0.75 * v6_probs + 0.25 * v3_probs
        else:
            probs = v6_probs

        # Apply EDA priors
        if not args.no_priors:
            probs = apply_color_priors(probs, img_rgb, orig_h, orig_w, hist_data=hist_data, eda_data=eda_data, joint_data=joint_data)

        pred = probs.argmax(axis=0).astype(np.uint8)

        # Update confusion matrix (skip invalid GT pixels)
        valid = (gt < NUM_CLASSES)
        gt_valid = gt[valid].astype(np.int64)
        pred_valid = pred[valid].astype(np.int64)
        indices = gt_valid * NUM_CLASSES + pred_valid
        confusion += np.bincount(indices, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (len(files) - idx - 1)
            print(f"  [{idx+1}/{len(files)}] {elapsed:.0f}s elapsed, ETA: {eta:.0f}s")

    # Compute metrics from confusion matrix
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(files):.2f}s/image)")

    print(f"\n{'='*60}")
    print(f"  FULL TEST SET RESULTS ({len(files)} images)")
    print(f"{'='*60}")

    # Pixel accuracy
    correct = np.diag(confusion).sum()
    total = confusion.sum()
    pixel_acc = correct / total if total > 0 else 0

    # Per-class IoU
    ious = []
    print(f"\n  {'Class':>16} | {'IoU':>8} | {'GT pixels':>12} | {'Pred pixels':>12}")
    print(f"  {'-'*60}")
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        gt_count = confusion[c, :].sum()
        pred_count = confusion[:, c].sum()

        if denom > 0:
            iou = float(tp / denom)
            ious.append(iou)
            bar = '█' * int(iou * 20)
            print(f"  {CLASS_NAMES[c]:>16} | {iou:>8.4f} | {gt_count:>12,} | {pred_count:>12,} {bar}")
        else:
            print(f"  {CLASS_NAMES[c]:>16} | {'N/A':>8} | {gt_count:>12,} | {pred_count:>12,}")

    miou = np.mean(ious) if ious else 0.0
    print(f"\n  {'mIoU':>16} | {miou:>8.4f}")
    print(f"  {'Pixel Acc':>16} | {pixel_acc:>8.4f}")
    print(f"  {'Classes eval':>16} | {len(ious):>8}")

    # Print confusion matrix summary (top confusions)
    print(f"\n  Top confusions (GT → Pred):")
    conf_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and confusion[i, j] > 0:
                conf_pairs.append((confusion[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    conf_pairs.sort(reverse=True)
    for count, gt_cls, pred_cls in conf_pairs[:10]:
        pct = count / total * 100
        print(f"    {gt_cls:>16} → {pred_cls:<16} : {count:>10,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
