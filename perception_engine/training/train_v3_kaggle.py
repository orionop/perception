# ============================================================
# PERCEPTION ENGINE V3 — OPTIMIZED TRAINING (6-hour Kaggle run)
# Target: 50-60% mIoU on 10-class offroad segmentation
# ============================================================
# CRITICAL FIXES from V2:
#   1. Augmentations applied to BOTH image AND mask (V2 only augmented image)
#   2. ImageNet normalization added (ResNet50 requires this)
#   3. 50 epochs + cosine LR scheduler (V2 had only 12 flat LR)
#   4. Encoder unfreezing after warmup (V2 never fine-tuned backbone)
#   5. Mixed precision training for 2x speed on T4
# ============================================================

# ---- CELL 1: SETUP ----
# !pip install -q segmentation-models-pytorch albumentations

import os, cv2, yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
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

IMG_HEIGHT = 288
IMG_WIDTH  = 512

def remap_mask(mask):
    """Convert raw pixel values (100, 200, ...) to class indices (0-9)."""
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask

print(f"Classes: {NUM_CLASSES}")
print(f"Input size: {IMG_HEIGHT}x{IMG_WIDTH}")


# ---- CELL 3: AUGMENTATIONS (APPLIED TO BOTH IMAGE AND MASK) ----
# FIX: V2 used torchvision transforms which only augment the image.
# Albumentations applies SAME spatial transforms to both image AND mask.

train_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    # Spatial (applied to BOTH image and mask):
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4,
                       border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, p=0.3),
    A.PadIfNeeded(IMG_HEIGHT, IMG_WIDTH, border_mode=cv2.BORDER_CONSTANT, value=0),
    # Color (image only — Albumentations handles this automatically):
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(5, 25), p=0.2),
    # Normalize with ImageNet stats (CRITICAL for pretrained ResNet50):
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

        # Load as RGB numpy array.
        image = cv2.imread(os.path.join(self.img_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask and remap to 0-9.
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Handle multi-channel masks
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = remap_mask(mask)

        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]       # Already tensor, normalized
            mask  = augmented["mask"].long()  # Already tensor
        
        return image, mask

train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_augment)
val_dataset   = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_augment)

# Bigger batch on T4 (16GB) with mixed precision
BATCH_SIZE = 12
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")


# ---- CELL 5: CLASS WEIGHTS ----
class_freq = np.array([0.0353, 0.0593, 0.1887, 0.0110, 0.0439,
                        0.0281, 0.0008, 0.0120, 0.2445, 0.3764])
# Inverse frequency, clipped to avoid extreme weights on log (0.08%)
weights = 1.0 / (class_freq + 1e-3)
weights = weights / weights.sum() * NUM_CLASSES
# Cap log class weight to prevent instability
weights = np.clip(weights, 0.5, 20.0)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("Class weights:", {n: f"{w:.2f}" for n, w in zip(CLASS_NAMES, weights)})


# ---- CELL 6: MODEL + LOSS + OPTIMIZER ----
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(device)

# Hybrid loss: Weighted CE + Dice (handles both class imbalance and boundary quality)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_oh = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def hybrid_loss(pred, target):
    return ce_loss_fn(pred, target) + dice_loss(pred, target)

# Phase 1: Freeze encoder, train decoder only (10 epochs)
# Phase 2: Unfreeze encoder, fine-tune everything (40 epochs)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

TOTAL_EPOCHS = 50
WARMUP_EPOCHS = 10

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)

# Mixed precision for faster training on T4
scaler = GradScaler()

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training plan: {WARMUP_EPOCHS} warmup (frozen encoder) + {TOTAL_EPOCHS - WARMUP_EPOCHS} fine-tune")


# ---- CELL 7: TRAINING LOOP ----
def compute_miou(preds, labels, num_classes=NUM_CLASSES):
    preds = preds.argmax(1)
    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = labels == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def compute_per_class_iou(preds, labels, num_classes=NUM_CLASSES):
    preds = preds.argmax(1)
    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = labels == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious

# --- Freeze encoder for warmup ---
for param in model.encoder.parameters():
    param.requires_grad = False
print("Encoder FROZEN for warmup phase")

best_miou = 0.0
history = []

for epoch in range(TOTAL_EPOCHS):
    # --- Unfreeze encoder after warmup ---
    if epoch == WARMUP_EPOCHS:
        for param in model.encoder.parameters():
            param.requires_grad = True
        # Reset optimizer with lower LR for encoder
        optimizer = optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": model.decoder.parameters(), "lr": 5e-4},
            {"params": model.segmentation_head.parameters(), "lr": 5e-4},
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )
        print(f"\n{'='*60}")
        print(f"ENCODER UNFROZEN — Fine-tuning all parameters")
        print(f"{'='*60}")

    # --- Train ---
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = hybrid_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    avg_loss = running_loss / len(train_loader)

    # --- Validate ---
    model.eval()
    val_miou = 0.0
    all_class_ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            with autocast():
                outputs = model(images)
            val_miou += compute_miou(outputs.cpu(), masks.cpu())
            all_class_ious.append(compute_per_class_iou(outputs.cpu(), masks.cpu()))

    val_miou /= len(val_loader)
    avg_class_ious = np.nanmean(all_class_ious, axis=0)

    lr = optimizer.param_groups[0]["lr"]
    phase = "WARMUP" if epoch < WARMUP_EPOCHS else "FINETUNE"

    print(f"\n[{phase}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {lr:.6f}")

    # Print per-class IoU every 5 epochs
    if (epoch + 1) % 5 == 0 or val_miou > best_miou:
        print("  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = avg_class_ious[i]
            bar = "█" * int(v * 20) if not np.isnan(v) else "N/A"
            val_str = f"{v:.3f}" if not np.isnan(v) else "N/A  "
            print(f"    {name:>16}: {val_str} {bar}")

    history.append({"epoch": epoch+1, "loss": avg_loss, "miou": val_miou})

    # Save best
    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), "best_model_v3.pth")
        print(f"  ★ NEW BEST: {best_miou:.4f} — saved best_model_v3.pth")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — Best mIoU: {best_miou:.4f}")
print(f"{'='*60}")


# ---- CELL 8: SAVE CONFIG ----
config = {
    "model_name": "deeplabv3plus_resnet50_v3",
    "architecture": "DeepLabV3Plus",
    "encoder": "resnet50",
    "num_classes": NUM_CLASSES,
    "input_size": [IMG_HEIGHT, IMG_WIDTH],
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
        "warmup_epochs": WARMUP_EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss": "CrossEntropy + Dice",
        "optimizer": "AdamW",
        "best_miou": float(best_miou),
    },
}

with open("config_v3.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config saved to config_v3.yaml")
print(f"Download: best_model_v3.pth + config_v3.yaml")
