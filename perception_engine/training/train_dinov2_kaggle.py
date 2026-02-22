# ============================================================
# PERCEPTION ENGINE V4 — DINOv2 ViT-B/14 TRAINING (Kaggle T4)
# ============================================================
# Architecture: Frozen DINOv2 ViT-B/14 backbone + ConvNeXt SegHead
# Target: 0.70-0.80 mIoU (Kaggle val), 0.55-0.65 (test)
#
# Why DINOv2:
#   - Self-supervised on 142M images → understands textures universally
#   - Frozen backbone → only ~3M trainable params → fast, no overfitting
#   - ~15-20% domain gap vs ~35%+ for ResNet/SegFormer
#
# Kaggle setup: GPU T4, Internet ON (for torch.hub.load)
# Runtime: ~3-4 hours for 40 epochs
# ============================================================

# ---- CELL 1: INSTALL + IMPORTS ----
# !pip install -q albumentations

import os, cv2, yaml, math
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---- CELL 2: PATHS + CLASS MAPPING ----
DATA_ROOT = "/kaggle/input/datasets/warwizardy/training-dataset/Offroad_Segmentation_Training_Dataset"

TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMG_DIR    = os.path.join(DATA_ROOT, "val", "Color_Images")
VAL_MASK_DIR   = os.path.join(DATA_ROOT, "val", "Segmentation")

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = len(CLASS_VALUES)

# DINOv2 requires input size divisible by patch_size (14)
# Original images are 960x540, half them and align to patch grid
IMG_WIDTH  = int(((960 / 2) // 14) * 14)   # = 476
IMG_HEIGHT = int(((540 / 2) // 14) * 14)   # = 266
PATCH_SIZE = 14
TOKEN_W = IMG_WIDTH  // PATCH_SIZE          # = 34
TOKEN_H = IMG_HEIGHT // PATCH_SIZE          # = 19

def remap_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask

print(f"Classes: {NUM_CLASSES}")
print(f"Input size: {IMG_HEIGHT}x{IMG_WIDTH} (patch-14 aligned)")
print(f"Token grid: {TOKEN_H}x{TOKEN_W} = {TOKEN_H * TOKEN_W} patches")


# ---- CELL 3: AUGMENTATIONS ----
# Strong augmentations for domain generalization
# Albumentations applies spatial transforms to BOTH image and mask

train_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    # Spatial (applied to both image AND mask):
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.4,
                       border_mode=cv2.BORDER_CONSTANT, value=0),
    # Color (image only — helps domain generalization):
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(var_limit=(5, 30)),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    # Normalize with ImageNet stats (required for DINOv2):
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ---- CELL 4: DATASET ----
class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = cv2.imread(os.path.join(self.img_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = remap_mask(mask)

        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].long()

        return image, mask

train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_augment)
val_dataset   = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_augment)

# Batch size 6 for ViT-B on T4 16GB (backbone features use more memory)
BATCH_SIZE = 6
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")


# ---- CELL 5: DINOV2 BACKBONE ----
print("Loading DINOv2 ViT-B/14 backbone...")
backbone = torch.hub.load(
    repo_or_dir="facebookresearch/dinov2",
    model="dinov2_vitb14_reg",   # ViT-Base with register tokens
)
backbone.eval()
backbone.to(device)

# Freeze ALL backbone parameters — we never train this
for param in backbone.parameters():
    param.requires_grad = False

# Get embedding dimension
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    dummy_out = backbone.forward_features(dummy)["x_norm_patchtokens"]
    EMBED_DIM = dummy_out.shape[2]
    n_patches = dummy_out.shape[1]

print(f"DINOv2 loaded: embed_dim={EMBED_DIM}, patches={n_patches}")
print(f"Expected patches: {TOKEN_H * TOKEN_W} = {TOKEN_H}x{TOKEN_W}")
assert n_patches == TOKEN_H * TOKEN_W, f"Patch count mismatch: {n_patches} vs {TOKEN_H * TOKEN_W}"
print("✅ Backbone verified!")


# ---- CELL 6: SEGMENTATION HEAD (ConvNeXt-style) ----
class SegmentationHead(nn.Module):
    """ConvNeXt-style segmentation head for DINOv2 patch tokens.
    
    Stronger than reference (256 channels, 2 blocks, BatchNorm)
    to better leverage ViT-B's 768-dim features.
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256  # Wider than reference's 128

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

        # Two ConvNeXt-style blocks for better feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.stem(x)
        x = x + self.block1(x)   # Residual connection
        x = x + self.block2(x)   # Residual connection
        return self.classifier(x)

seg_head = SegmentationHead(
    in_channels=EMBED_DIM,
    out_channels=NUM_CLASSES,
    tokenW=TOKEN_W,
    tokenH=TOKEN_H,
).to(device)

trainable_params = sum(p.numel() for p in seg_head.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in backbone.parameters()) + trainable_params
print(f"Segmentation head: {trainable_params:,} trainable params")
print(f"Total model: {total_params:,} params ({trainable_params:,} trainable)")


# ---- CELL 7: LOSS + OPTIMIZER ----
# Class weights (inverse sqrt frequency, same as V3)
class_freq = np.array([0.0353, 0.0593, 0.1887, 0.0110, 0.0439,
                        0.0281, 0.0008, 0.0120, 0.2445, 0.3764])
weights = 1.0 / np.sqrt(class_freq)
weights = weights / weights.min()
weights = np.clip(weights, 1.0, 15.0)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("Class weights:", {n: f"{w:.1f}" for n, w in zip(CLASS_NAMES, weights)})

ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def hybrid_loss(pred, target):
    # Upsample predictions to match target size
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode="bilinear", align_corners=False)
    return ce_loss_fn(pred, target) + dice_loss(pred, target)

# Only optimize the segmentation head (backbone is frozen)
optimizer = optim.AdamW(seg_head.parameters(), lr=2e-3, weight_decay=1e-4)

TOTAL_EPOCHS = 40
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

scaler = GradScaler()
print(f"Training plan: {TOTAL_EPOCHS} epochs, head only (backbone frozen)")


# ---- CELL 8: TRAINING LOOP ----
def compute_miou(preds, labels):
    preds = preds.argmax(1)
    ious = []
    for cls in range(NUM_CLASSES):
        pred_c = preds == cls
        target_c = labels == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def compute_per_class_iou(preds, labels):
    preds = preds.argmax(1)
    ious = []
    for cls in range(NUM_CLASSES):
        pred_c = preds == cls
        target_c = labels == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious

best_miou = 0.0
history = []

for epoch in range(TOTAL_EPOCHS):
    # --- Train (only seg_head, backbone is frozen) ---
    seg_head.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        # Extract DINOv2 features (no gradient needed)
        with torch.no_grad():
            features = backbone.forward_features(images)["x_norm_patchtokens"]

        # Forward through trainable head
        with autocast():
            logits = seg_head(features)
            loss = hybrid_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # --- Validate ---
    seg_head.eval()
    val_miou = 0.0
    all_class_ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            features = backbone.forward_features(images)["x_norm_patchtokens"]
            with autocast():
                logits = seg_head(features)
                # Upsample to mask resolution for metric computation
                logits = F.interpolate(logits, size=masks.shape[1:],
                                       mode="bilinear", align_corners=False)

            val_miou += compute_miou(logits.cpu(), masks.cpu())
            all_class_ious.append(compute_per_class_iou(logits.cpu(), masks.cpu()))

    val_miou /= len(val_loader)
    avg_class_ious = np.nanmean(all_class_ious, axis=0)
    lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {lr:.6f}")

    # Print per-class IoU every 5 epochs or on new best
    if (epoch + 1) % 5 == 0 or val_miou > best_miou:
        print("  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = avg_class_ious[i]
            bar = "█" * int(v * 20) if not np.isnan(v) else ""
            val_str = f"{v:.3f}" if not np.isnan(v) else "N/A  "
            print(f"    {name:>16}: {val_str} {bar}")

    history.append({"epoch": epoch+1, "loss": avg_loss, "miou": val_miou})

    if val_miou > best_miou:
        best_miou = val_miou
        # Save only the seg_head weights (backbone is standard DINOv2)
        torch.save(seg_head.state_dict(), "best_dinov2_head.pth")
        # Also save full inference state for easy loading
        torch.save({
            "seg_head": seg_head.state_dict(),
            "backbone_name": "dinov2_vitb14_reg",
            "embed_dim": EMBED_DIM,
            "token_w": TOKEN_W,
            "token_h": TOKEN_H,
            "num_classes": NUM_CLASSES,
            "img_size": [IMG_HEIGHT, IMG_WIDTH],
        }, "best_model_v4.pth")
        print(f"  ★ NEW BEST: {best_miou:.4f} — saved best_model_v4.pth")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — Best mIoU: {best_miou:.4f}")
print(f"{'='*60}")


# ---- CELL 9: SAVE CONFIG ----
config = {
    "model_name": "dinov2_vitb14_v4",
    "architecture": "DINOv2_ConvNeXtHead",
    "backbone": "dinov2_vitb14_reg",
    "encoder": "vitb14_reg",
    "embed_dim": EMBED_DIM,
    "num_classes": NUM_CLASSES,
    "input_size": [IMG_HEIGHT, IMG_WIDTH],
    "patch_size": PATCH_SIZE,
    "token_grid": [TOKEN_H, TOKEN_W],
    "class_names": CLASS_NAMES,
    "mask_value_mapping": {str(k): v for k, v in CLASS_TO_INDEX.items()},
    "preprocessing": {
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
        "target_size": [IMG_HEIGHT, IMG_WIDTH],
    },
    "training": {
        "epochs": TOTAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss": "CrossEntropy + Dice",
        "optimizer": "AdamW",
        "backbone_frozen": True,
        "best_miou": float(best_miou),
    },
}

with open("config_v4.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config saved to config_v4.yaml")
print(f"\nDownload these files:")
print(f"  1. best_model_v4.pth  (seg head weights + metadata)")
print(f"  2. config_v4.yaml     (full config for perception engine)")
