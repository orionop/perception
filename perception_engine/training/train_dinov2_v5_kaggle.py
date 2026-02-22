# ============================================================
# PERCEPTION ENGINE V5 — DINOv2 ViT-B/14 UNFROZEN (Kaggle T4)
# ============================================================
# Key change from V4: UNFREEZE last 4 transformer blocks after warmup
# This is the single biggest lever for accuracy.
#
# Schedule:
#   Epochs 1-5:   Backbone FROZEN, train head only (warmup)
#   Epoch 5:      ONE cosine restart (shake out of local minimum)
#   Epochs 6-10:  Still frozen, head converges
#   Epoch 11:     UNFREEZE last 4 DINOv2 blocks
#   Epochs 11-60: Fine-tune backbone (LR=2e-5) + head (LR=5e-4)
#                 Plain cosine decay to 1e-6, NO MORE RESTARTS
#
# Target: 0.72-0.80 Kaggle val, 0.60-0.68 test
# Runtime: ~4-5 hours on T4
# ============================================================

# ---- CELL 1: INSTALL + IMPORTS ----
# !pip install -q albumentations

import os, cv2, yaml, math, gc
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

# DINOv2 requires input divisible by patch_size=14
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
print(f"Input: {IMG_HEIGHT}x{IMG_WIDTH}, Patches: {TOKEN_H}x{TOKEN_W}")


# ---- CELL 3: STRONG AUGMENTATIONS ----
train_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.4,
                       border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(var_limit=(5, 30)),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
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

# Batch size 4 when backbone unfrozen (more memory needed for gradients)
BATCH_SIZE_FROZEN = 6
BATCH_SIZE_UNFROZEN = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_FROZEN, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE_FROZEN, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")


# ---- CELL 5: DINOV2 BACKBONE ----
print("Loading DINOv2 ViT-B/14...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
backbone.eval()
backbone.to(device)

# Freeze everything initially
for param in backbone.parameters():
    param.requires_grad = False

# Get embedding dimension
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    dummy_out = backbone.forward_features(dummy)["x_norm_patchtokens"]
    EMBED_DIM = dummy_out.shape[2]
del dummy, dummy_out
torch.cuda.empty_cache()

print(f"DINOv2 loaded: embed_dim={EMBED_DIM}, tokens={TOKEN_H}x{TOKEN_W}")
print(f"Backbone blocks: {len(backbone.blocks)}")


# ---- CELL 6: SEGMENTATION HEAD ----
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
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
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.classifier(x)

seg_head = SegmentationHead(EMBED_DIM, NUM_CLASSES, TOKEN_W, TOKEN_H).to(device)

print(f"Head params: {sum(p.numel() for p in seg_head.parameters()):,}")


# ---- CELL 7: LOSS ----
class_freq = np.array([0.0353, 0.0593, 0.1887, 0.0110, 0.0439,
                        0.0281, 0.0008, 0.0120, 0.2445, 0.3764])
weights = 1.0 / np.sqrt(class_freq)
weights = weights / weights.min()
weights = np.clip(weights, 1.0, 15.0)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def hybrid_loss(pred, target):
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode="bilinear", align_corners=False)
    return ce_loss_fn(pred, target) + dice_loss(pred, target)


# ---- CELL 8: METRICS ----
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


# ---- CELL 9: TRAINING ----
WARMUP_EPOCHS = 10       # Frozen backbone
UNFREEZE_EPOCH = 11      # Unfreeze at this epoch
TOTAL_EPOCHS = 60        # Total training
UNFREEZE_BLOCKS = 4      # Number of last transformer blocks to unfreeze

# Phase 1: Head-only optimizer
optimizer = optim.AdamW(seg_head.parameters(), lr=2e-3, weight_decay=1e-4)

# ONE restart at epoch 5, then done
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=100, eta_min=1e-5  # T_mult=100 means no second restart
)

scaler = GradScaler()
best_miou = 0.0
history = []

print(f"Phase 1: FROZEN backbone, epochs 1-{WARMUP_EPOCHS}")
print(f"Phase 2: UNFREEZE last {UNFREEZE_BLOCKS} blocks, epochs {UNFREEZE_EPOCH}-{TOTAL_EPOCHS}")
print(f"One LR restart at epoch 5, then plain cosine decay")
print(f"{'='*60}\n")

for epoch in range(TOTAL_EPOCHS):

    # === UNFREEZE at epoch 11 ===
    if epoch + 1 == UNFREEZE_EPOCH:
        print(f"\n{'='*60}")
        print(f"UNFREEZING last {UNFREEZE_BLOCKS} DINOv2 blocks!")
        print(f"{'='*60}\n")

        # Unfreeze last N transformer blocks
        for block in backbone.blocks[-UNFREEZE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze the final norm layer
        if hasattr(backbone, 'norm'):
            for param in backbone.norm.parameters():
                param.requires_grad = True

        unfrozen_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Unfrozen backbone params: {unfrozen_params:,}")

        # New optimizer with differential LR
        optimizer = optim.AdamW([
            {"params": seg_head.parameters(), "lr": 5e-4},
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 2e-5},
        ], weight_decay=1e-4)

        # Plain cosine decay for the rest — NO RESTARTS
        remaining = TOTAL_EPOCHS - UNFREEZE_EPOCH + 1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-6
        )

        # Reduce batch size (backbone gradients use more memory)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_UNFROZEN,
                                  shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_UNFROZEN,
                                shuffle=False, num_workers=2, pin_memory=True)

        # Clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()

    # === Train ===
    seg_head.train()
    if epoch + 1 >= UNFREEZE_EPOCH:
        backbone.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if epoch + 1 < UNFREEZE_EPOCH:
            # Frozen: no grad for backbone
            with torch.no_grad():
                features = backbone.forward_features(images)["x_norm_patchtokens"]
            with autocast():
                logits = seg_head(features)
                loss = hybrid_loss(logits, masks)
        else:
            # Unfrozen: grad through backbone
            with autocast():
                features = backbone.forward_features(images)["x_norm_patchtokens"]
                logits = seg_head(features)
                loss = hybrid_loss(logits, masks)

        scaler.scale(loss).backward()

        # Gradient clipping for stability when fine-tuning backbone
        if epoch + 1 >= UNFREEZE_EPOCH:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(seg_head.parameters()) + [p for p in backbone.parameters() if p.requires_grad],
                max_norm=1.0
            )

        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # === Validate ===
    seg_head.eval()
    backbone.eval()
    val_miou = 0.0
    all_class_ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            features = backbone.forward_features(images)["x_norm_patchtokens"]
            with autocast():
                logits = seg_head(features)
                logits = F.interpolate(logits, size=masks.shape[1:],
                                       mode="bilinear", align_corners=False)
            val_miou += compute_miou(logits.cpu(), masks.cpu())
            all_class_ious.append(compute_per_class_iou(logits.cpu(), masks.cpu()))

    val_miou /= len(val_loader)
    avg_class_ious = np.nanmean(all_class_ious, axis=0)

    phase = "FROZEN" if epoch + 1 <= WARMUP_EPOCHS else "FINETUNE"
    head_lr = optimizer.param_groups[0]["lr"]
    backbone_lr = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

    print(f"\n[{phase}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} | "
          f"Head LR: {head_lr:.6f} | Backbone LR: {backbone_lr:.6f}")

    if (epoch + 1) % 5 == 0 or val_miou > best_miou:
        print("  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = avg_class_ious[i]
            bar = "█" * int(v * 20) if not np.isnan(v) else ""
            val_str = f"{v:.3f}" if not np.isnan(v) else "N/A  "
            print(f"    {name:>16}: {val_str} {bar}")

    history.append({"epoch": epoch+1, "loss": avg_loss, "miou": val_miou, "phase": phase})

    if val_miou > best_miou:
        best_miou = val_miou

        # Save full checkpoint (backbone blocks + head)
        save_dict = {
            "seg_head": seg_head.state_dict(),
            "backbone_name": "dinov2_vitb14_reg",
            "embed_dim": EMBED_DIM,
            "token_w": TOKEN_W,
            "token_h": TOKEN_H,
            "num_classes": NUM_CLASSES,
            "img_size": [IMG_HEIGHT, IMG_WIDTH],
            "epoch": epoch + 1,
            "miou": val_miou,
        }

        # Save backbone blocks that were fine-tuned
        if epoch + 1 >= UNFREEZE_EPOCH:
            save_dict["backbone_blocks"] = {}
            for i, block in enumerate(backbone.blocks[-UNFREEZE_BLOCKS:]):
                block_idx = len(backbone.blocks) - UNFREEZE_BLOCKS + i
                save_dict["backbone_blocks"][f"block_{block_idx}"] = block.state_dict()
            if hasattr(backbone, 'norm'):
                save_dict["backbone_norm"] = backbone.norm.state_dict()

        torch.save(save_dict, "best_model_v5.pth")
        print(f"  ★ NEW BEST: {best_miou:.4f} — saved best_model_v5.pth")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — Best mIoU: {best_miou:.4f}")
print(f"{'='*60}")


# ---- CELL 10: SAVE CONFIG ----
config = {
    "model_name": "dinov2_vitb14_v5",
    "architecture": "DINOv2_ConvNeXtHead",
    "backbone": "dinov2_vitb14_reg",
    "encoder": "vitb14_reg",
    "embed_dim": EMBED_DIM,
    "num_classes": NUM_CLASSES,
    "input_size": [IMG_HEIGHT, IMG_WIDTH],
    "patch_size": PATCH_SIZE,
    "token_grid": [TOKEN_H, TOKEN_W],
    "unfrozen_blocks": UNFREEZE_BLOCKS,
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
        "total_epochs": TOTAL_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "unfreeze_epoch": UNFREEZE_EPOCH,
        "batch_size_frozen": BATCH_SIZE_FROZEN,
        "batch_size_unfrozen": BATCH_SIZE_UNFROZEN,
        "loss": "CrossEntropy + Dice",
        "optimizer": "AdamW",
        "best_miou": float(best_miou),
    },
}

with open("config_v5.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Saved: config_v5.yaml")
print(f"\nDownload: best_model_v5.pth + config_v5.yaml")
