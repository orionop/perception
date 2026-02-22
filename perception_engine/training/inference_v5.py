"""
Inference script for V5 model (DINOv2 + ConvNeXt Head).
Loads best_model_v5.pth, runs on test images, saves colored segmentation masks.
"""

import os, sys, cv2, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ---- CONFIG ----
# Paths relative to project root (parent of perception_engine)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
CHECKPOINT = os.path.join(_PROJECT_ROOT, "weights", "best_model_v5.pth")
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "perception_engine", "configs", "config_v5.yaml")
TEST_IMG_DIR = os.path.join(_PROJECT_ROOT, "perception_engine", "Offroad_Segmentation_testImages", "Color_Images")
TEST_MASK_DIR = os.path.join(_PROJECT_ROOT, "perception_engine", "Offroad_Segmentation_testImages", "Segmentation")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "v5_predictions")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

IMG_HEIGHT, IMG_WIDTH = config["input_size"]
TOKEN_H, TOKEN_W = config["token_grid"]
NUM_CLASSES = config["num_classes"]
CLASS_NAMES = config["class_names"]
EMBED_DIM = config["embed_dim"]
UNFROZEN_BLOCKS = config["unfrozen_blocks"]
MEAN = config["preprocessing"]["normalize"]["mean"]
STD = config["preprocessing"]["normalize"]["std"]

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}

def remap_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask

CLASS_COLORS = np.array([
    [34, 139, 34],    # tree — forest green
    [0, 200, 0],      # lush_bush — bright green
    [210, 180, 80],   # dry_grass — tan/gold
    [139, 119, 42],   # dry_bush — dark khaki
    [128, 128, 128],  # ground_clutter — gray
    [255, 105, 180],  # flower — hot pink
    [139, 69, 19],    # log — saddle brown
    [105, 105, 105],  # rock — dim gray
    [210, 180, 140],  # landscape — tan
    [135, 206, 235],  # sky — sky blue
], dtype=np.uint8)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


# ---- MODEL DEFINITION (must match V5 training) ----
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU())
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(hidden, out_channels, 1))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.classifier(x)


# ---- LOAD BACKBONE ----
print("Loading DINOv2 ViT-B/14...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
backbone.eval()
backbone.to(device)

# ---- LOAD CHECKPOINT ----
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

seg_head = SegmentationHead(EMBED_DIM, NUM_CLASSES, TOKEN_W, TOKEN_H).to(device)
seg_head.load_state_dict(ckpt["seg_head"])
seg_head.eval()

if "backbone_blocks" in ckpt:
    N_BLOCKS = len(backbone.blocks)
    loaded = 0
    for bname, bstate in ckpt["backbone_blocks"].items():
        idx = int(bname.split("_")[1])
        backbone.blocks[idx].load_state_dict(bstate)
        loaded += 1
    print(f"  Loaded {loaded} fine-tuned backbone blocks")
if "backbone_norm" in ckpt:
    backbone.norm.load_state_dict(ckpt["backbone_norm"])
    print("  Loaded backbone norm")

print(f"  Checkpoint mIoU: {ckpt.get('miou', '?'):.4f}, epoch: {ckpt.get('epoch', '?')}")

for param in backbone.parameters():
    param.requires_grad = False
for param in seg_head.parameters():
    param.requires_grad = False


# ---- PREPROCESSING ----
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        color[mask == cls_id] = CLASS_COLORS[cls_id]
    return color


def blend_overlay(image, mask_color, alpha=0.5):
    image_resized = cv2.resize(image, (mask_color.shape[1], mask_color.shape[0]))
    return cv2.addWeighted(image_resized, 1 - alpha, mask_color, alpha, 0)


# ---- INFERENCE ----
@torch.no_grad()
def predict(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(image_rgb).unsqueeze(0).to(device)

    features = backbone.forward_features(tensor)["x_norm_patchtokens"]
    logits = seg_head(features)
    logits = F.interpolate(logits, size=(IMG_HEIGHT, IMG_WIDTH), mode="bilinear", align_corners=False)

    # Flip TTA
    tensor_f = torch.flip(tensor, dims=[3])
    features_f = backbone.forward_features(tensor_f)["x_norm_patchtokens"]
    logits_f = seg_head(features_f)
    logits_f = F.interpolate(logits_f, size=(IMG_HEIGHT, IMG_WIDTH), mode="bilinear", align_corners=False)
    logits_f = torch.flip(logits_f, dims=[3])

    logits_avg = (logits + logits_f) / 2.0
    pred = logits_avg.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    conf = torch.softmax(logits_avg, dim=1).max(1).values.squeeze(0).cpu().numpy()
    return pred, conf


# ---- MAIN ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--all", action="store_true", help="Run on all test images")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    overlay_dir = os.path.join(OUTPUT_DIR, "overlays")
    conf_dir = os.path.join(OUTPUT_DIR, "confidence")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    if args.image:
        files = [os.path.basename(args.image)]
        img_dir = os.path.dirname(args.image)
    elif args.all:
        files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')])
        img_dir = TEST_IMG_DIR
    else:
        files = ["0000156.png"]
        img_dir = TEST_IMG_DIR

    print(f"\nRunning inference on {len(files)} image(s)...")
    print(f"Output: {OUTPUT_DIR}\n")

    # mIoU tracking
    global_intersection = np.zeros(NUM_CLASSES, dtype=np.int64)
    global_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    has_gt = os.path.isdir(TEST_MASK_DIR)
    evaluated = 0

    for i, fname in enumerate(files):
        path = os.path.join(img_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname} not found")
            continue

        image_bgr = cv2.imread(path)
        pred_mask, confidence = predict(image_bgr)

        # Evaluate against GT if available
        if has_gt:
            gt_path = os.path.join(TEST_MASK_DIR, fname)
            if os.path.exists(gt_path):
                gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if gt_raw is not None:
                    if len(gt_raw.shape) == 3:
                        gt_raw = gt_raw[:, :, 0]
                    gt_mask = remap_mask(gt_raw)
                    gt_resized = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    for cls in range(NUM_CLASSES):
                        pred_c = pred_mask == cls
                        gt_c = gt_resized == cls
                        global_intersection[cls] += int((pred_c & gt_c).sum())
                        global_union[cls] += int((pred_c | gt_c).sum())
                    evaluated += 1

        mask_color = colorize_mask(pred_mask)
        overlay = blend_overlay(image_bgr, mask_color, alpha=0.45)
        conf_vis = (confidence * 255).astype(np.uint8)
        conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(mask_dir, f"{base}_mask.png"), cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(overlay_dir, f"{base}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(conf_dir, f"{base}_confidence.png"), conf_colored)

        if (i + 1) % 100 == 0 or i == 0 or len(files) == 1:
            mean_conf = float(confidence.mean())
            print(f"  [{i+1}/{len(files)}] {fname} | conf={mean_conf:.3f}")

    # ---- FINAL mIoU REPORT ----
    if evaluated > 0:
        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION — {evaluated} images")
        print(f"{'='*60}")
        per_class_iou = []
        for cls in range(NUM_CLASSES):
            if global_union[cls] > 0:
                iou = global_intersection[cls] / global_union[cls]
            else:
                iou = float('nan')
            per_class_iou.append(iou)
            bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
            val_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A  "
            print(f"  {CLASS_NAMES[cls]:>16}: {val_str} {bar}")

        valid_ious = [v for v in per_class_iou if not np.isnan(v)]
        test_miou = np.mean(valid_ious) if valid_ious else 0.0
        print(f"\n  *** TEST mIoU: {test_miou:.4f} ***")
        print(f"{'='*60}")
    else:
        # Per-class pixel stats for single image
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_px = pred_mask.size
        for cls_id, cnt in zip(unique, counts):
            pct = cnt / total_px * 100
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            bar = "█" * int(pct / 2)
            print(f"    {name:>16}: {pct:5.1f}% {bar}")

    # Save legend
    legend_h = 30 * NUM_CLASSES + 20
    legend = np.ones((legend_h, 300, 3), dtype=np.uint8) * 30
    for i, name in enumerate(CLASS_NAMES):
        y = 10 + i * 30
        cv2.rectangle(legend, (10, y), (35, y + 20), CLASS_COLORS[i].tolist(), -1)
        cv2.putText(legend, name, (45, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "legend.png"), legend)

    print(f"\nDone! Outputs saved to {OUTPUT_DIR}/")
    print(f"  masks/     — colored segmentation masks")
    print(f"  overlays/  — original + mask blended")
    print(f"  confidence/ — per-pixel confidence heatmaps")
