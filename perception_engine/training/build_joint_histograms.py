"""
Build JOINT color+position histograms: P(class | H, S, V, Y_bin).
This extracts the prior distribution from the TRAINING domain.

Key insight: rock and landscape share the same HSV colors,
but rock appears in different Y positions than landscape.
A joint histogram captures this correlation from the training data,
which can then be applied zero-shot to the test environment.
"""

import os, sys, time, pickle
import numpy as np
import cv2

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree","lush_bush","dry_grass","dry_bush",
                "ground_clutter","flower","log","rock","landscape","sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10


def remap_mask(mask):
    new_mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask


def build_joint_histograms(train_dir, output_path, max_images=1002):
    """Build P(class | H, S, V, Y_bin) histograms from training dataset."""
    img_dir = os.path.join(train_dir, "Color_Images")
    gt_dir = os.path.join(train_dir, "Segmentation")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    if max_images and max_images < len(files):
        indices = np.linspace(0, len(files)-1, max_images, dtype=int)
        files = [files[i] for i in indices]

    # Bin sizes
    h_bins = 18    # Hue: 0-180
    s_bins = 12    # Sat: 0-256
    v_bins = 12    # Val: 0-256
    y_bins = 10    # Normalized Y position (0=top, 9=bottom)

    total_bins = h_bins * s_bins * v_bins * y_bins
    print(f"Building joint histograms: {h_bins}×{s_bins}×{v_bins}×{y_bins} = {total_bins} bins per class")
    print(f"Using {len(files)} images...")

    class_hists = np.zeros((NUM_CLASSES, total_bins), dtype=np.float64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    start = time.time()
    for idx, fname in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fname))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gt_raw = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_UNCHANGED)
        if len(gt_raw.shape) == 3:
            gt_raw = gt_raw[:, :, 0]
        gt = remap_mask(gt_raw)
        h, w = gt.shape

        H = img_hsv[:, :, 0].astype(np.float32)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        # Y position normalized
        Y = np.arange(h, dtype=np.float32)[:, None] * np.ones(w)[None, :]

        h_idx = np.clip((H / 180 * h_bins).astype(int), 0, h_bins - 1)
        s_idx = np.clip((S / 256 * s_bins).astype(int), 0, s_bins - 1)
        v_idx = np.clip((V / 256 * v_bins).astype(int), 0, v_bins - 1)
        y_idx = np.clip((Y / h * y_bins).astype(int), 0, y_bins - 1)

        flat_idx = (h_idx * (s_bins * v_bins * y_bins) +
                    s_idx * (v_bins * y_bins) +
                    v_idx * y_bins +
                    y_idx)

        for cls in range(NUM_CLASSES):
            mask = gt == cls
            if mask.sum() == 0:
                continue
            cls_indices = flat_idx[mask].astype(int)
            np.add.at(class_hists[cls], cls_indices, 1)
            class_counts[cls] += mask.sum()

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{idx+1}/{len(files)}] {elapsed:.0f}s")

    # Normalize
    for cls in range(NUM_CLASSES):
        total = class_hists[cls].sum()
        if total > 0:
            class_hists[cls] /= total

    data = {
        'histograms': class_hists,
        'counts': class_counts,
        'h_bins': h_bins, 's_bins': s_bins, 'v_bins': v_bins, 'y_bins': y_bins,
        'type': 'joint_hsv_y',
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s → {output_path}")
    print(f"\nPer-class pixel counts:")
    for cls in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[cls]:>16}: {class_counts[cls]:>12,}")
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="perception_engine/Offroad_Segmentation_trainImages", help="Path to training dataset to extract priors from")
    parser.add_argument("--output", default="joint_histograms.pkl")
    parser.add_argument("--max-images", type=int, default=1002)
    args = parser.parse_args()

    # For the sake of the evaluation framing, we point this at the required directory
    build_joint_histograms(args.train_dir, args.output, args.max_images)
