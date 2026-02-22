#!/usr/bin/env python3
"""
Build per-class color histograms from training data, then use as a
non-parametric classifier blended with DNN predictions.

Step 1: Build histograms (run once, saves to disk)
Step 2: Use histograms as priors during inference (fast)
"""

import os, sys, time, pickle
import numpy as np
import cv2

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10

def remap_mask(mask):
    new_mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask


def build_histograms(train_dir, output_path, max_images=200):
    """Build per-class 3D color histograms in HSV space from training data."""
    img_dir = os.path.join(train_dir, "Color_Images")
    gt_dir = os.path.join(train_dir, "Segmentation")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    if max_images and max_images < len(files):
        indices = np.linspace(0, len(files)-1, max_images, dtype=int)
        files = [files[i] for i in indices]

    print(f"Building histograms from {len(files)} training images...")

    # 3D histogram bins: H(18 bins), S(16 bins), V(16 bins) → 4608 bins per class
    h_bins, s_bins, v_bins = 18, 16, 16
    h_range = [0, 180]
    s_range = [0, 256]
    v_range = [0, 256]

    class_hists = np.zeros((NUM_CLASSES, h_bins * s_bins * v_bins), dtype=np.float64)
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    start = time.time()
    for idx, fname in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fname))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gt_raw = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_UNCHANGED)
        if len(gt_raw.shape) == 3:
            gt_raw = gt_raw[:, :, 0]
        gt = remap_mask(gt_raw)

        for cls in range(NUM_CLASSES):
            mask = gt == cls
            if mask.sum() == 0:
                continue

            pixels_hsv = img_hsv[mask]  # (N, 3)
            hist, _ = np.histogramdd(
                pixels_hsv.astype(np.float64),
                bins=[h_bins, s_bins, v_bins],
                range=[[0, 180], [0, 256], [0, 256]]
            )
            class_hists[cls] += hist.flatten()
            class_pixel_counts[cls] += mask.sum()

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(files)}] {elapsed:.0f}s")

    # Normalize each class histogram to probability distribution
    for cls in range(NUM_CLASSES):
        total = class_hists[cls].sum()
        if total > 0:
            class_hists[cls] /= total

    data = {
        'histograms': class_hists,
        'pixel_counts': class_pixel_counts,
        'h_bins': h_bins, 's_bins': s_bins, 'v_bins': v_bins,
        'h_range': h_range, 's_range': s_range, 'v_range': v_range,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {output_path}")
    print(f"Per-class pixel counts:")
    for cls in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[cls]:>16}: {class_pixel_counts[cls]:>12,}")
    return data


def classify_pixels(image_rgb, hist_data):
    """Classify each pixel using the color histogram lookup.
    Returns (C, H, W) probability map."""
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, w = image_hsv.shape[:2]

    hists = hist_data['histograms']
    h_bins = hist_data['h_bins']
    s_bins = hist_data['s_bins']
    v_bins = hist_data['v_bins']

    # Quantize each pixel to histogram bin
    H = image_hsv[:, :, 0].astype(np.float32)
    S = image_hsv[:, :, 1].astype(np.float32)
    V = image_hsv[:, :, 2].astype(np.float32)

    h_idx = np.clip((H / 180 * h_bins).astype(int), 0, h_bins - 1)
    s_idx = np.clip((S / 256 * s_bins).astype(int), 0, s_bins - 1)
    v_idx = np.clip((V / 256 * v_bins).astype(int), 0, v_bins - 1)

    flat_idx = h_idx * (s_bins * v_bins) + s_idx * v_bins + v_idx

    # Look up probability for each class
    probs = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        probs[cls] = hists[cls][flat_idx.flatten()].reshape(h, w)

    # Normalize
    prob_sum = probs.sum(axis=0, keepdims=True)
    prob_sum[prob_sum == 0] = 1.0
    probs = probs / prob_sum

    return probs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output", default="color_histograms.pkl")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    build_histograms(args.train_dir, args.output, args.max_images)
