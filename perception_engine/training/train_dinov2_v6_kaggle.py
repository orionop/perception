# ============================================================
# PERCEPTION ENGINE V6 — DOMAIN-ROBUST DINOv2 (FINAL)
# ============================================================
# THE FIX: Extreme domain randomization forces the model to learn
# STRUCTURE (shape, texture patterns, edges) instead of COLOR
# (which changes between training and test environments).
#
# Key additions over V5:
#   1. FDA (Fourier Domain Adaptation) — swaps low-frequency spectrum
#      between images, simulating different lighting/color conditions
#   2. CutMix — pastes random patches between images, forces robust
#      feature learning at boundaries
#   3. Extreme color augmentation — model can't rely on "green=grass"
#   4. Random channel shuffle — breaks color-class associations
#   5. CLAHE + histogram equalization — normalize contrast across domains
#   6. Longer training (80 epochs) with proper unfreezing
#
# Schedule:
#   Epochs 1-8:    Backbone FROZEN, train head only
#   Epoch 5:       ONE warm restart
#   Epoch 9:       UNFREEZE last 6 blocks (more than V5's 4)
#   Epochs 9-80:   Fine-tune with plain cosine decay, NO MORE RESTARTS
#
# Target: 0.75+ Kaggle val → 0.60+ test (domain-robust)
# ============================================================

# ---- CELL 1: INSTALL + IMPORTS ----
# !pip install -q albumentations

import os, cv2, yaml, gc, random
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
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---- CELL 2: CONFIG ----
DATA_ROOT = "/kaggle/input/datasets/warwizardy/training-dataset/Offroad_Segmentation_Training_Dataset"

TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMG_DIR    = os.path.join(DATA_ROOT, "val", "Color_Images")
VAL_MASK_DIR   = os.path.join(DATA_ROOT, "val", "Segmentation")

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10

# DINOv2 patch-14 aligned
IMG_WIDTH  = int(((960 / 2) // 14) * 14)   # 476
IMG_HEIGHT = int(((540 / 2) // 14) * 14)   # 266
PATCH_SIZE = 14
TOKEN_W = IMG_WIDTH  // PATCH_SIZE          # 34
TOKEN_H = IMG_HEIGHT // PATCH_SIZE          # 19

def remap_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask


# ---- CELL 3: DOMAIN RANDOMIZATION AUGMENTATIONS ----
class FourierDomainAdaptation(A.ImageOnlyTransform):
    """Swap low-frequency components to simulate different lighting/color."""
    def __init__(self, beta=0.01, always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.beta = beta
    
    def apply(self, img, **params):
        h, w = img.shape[:2]
        target = np.random.randint(0, 256, img.shape, dtype=np.uint8)
        result = np.zeros_like(img)
        for c in range(3):
            fft_src = np.fft.fft2(img[:,:,c].astype(np.float32))
            fft_trg = np.fft.fft2(target[:,:,c].astype(np.float32))
            fft_src_s = np.fft.fftshift(fft_src)
            fft_trg_s = np.fft.fftshift(fft_trg)
            amp_src = np.abs(fft_src_s)
            pha_src = np.angle(fft_src_s)
            amp_trg = np.abs(fft_trg_s)
            cy, cx = h // 2, w // 2
            b = int(min(h, w) * self.beta)
            if b > 0:
                amp_src[cy-b:cy+b, cx-b:cx+b] = (
                    0.5 * amp_src[cy-b:cy+b, cx-b:cx+b] +
                    0.5 * amp_trg[cy-b:cy+b, cx-b:cx+b]
                )
            fft_new = amp_src * np.exp(1j * pha_src)
            result[:,:,c] = np.real(np.fft.ifft2(np.fft.ifftshift(fft_new)))
        return np.clip(result, 0, 255).astype(np.uint8)


class ChannelShuffle(A.ImageOnlyTransform):
    """Randomly shuffle RGB channels — breaks color-class associations."""
    def __init__(self, always_apply=False, p=0.2):
        super().__init__(always_apply, p)
    def apply(self, img, **params):
        channels = list(range(3))
        random.shuffle(channels)
        return img[:, :, channels]


class GrayscaleAug(A.ImageOnlyTransform):
    """Forces model to use texture/shape, not color."""
    def __init__(self, always_apply=False, p=0.1):
        super().__init__(always_apply, p)
    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


train_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    
    # Spatial (both image AND mask)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5,
                       border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, p=0.3),
    A.PadIfNeeded(IMG_HEIGHT, IMG_WIDTH, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ElasticTransform(alpha=30, sigma=5, p=0.15),
    A.GridDistortion(p=0.15),
    
    # Color domain randomization (image only) — AGGRESSIVE
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
    
    A.OneOf([
        A.CLAHE(clip_limit=4.0, p=1.0),
        A.Equalize(p=1.0),
    ], p=0.3),
    
    FourierDomainAdaptation(beta=0.01, p=0.2),
    ChannelShuffle(p=0.15),
    GrayscaleAug(p=0.1),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.GaussNoise(var_limit=(10, 50)),
        A.ISONoise(p=1.0),
    ], p=0.3),
    
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
    
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

BATCH_SIZE_FROZEN = 6
BATCH_SIZE_UNFROZEN = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_FROZEN, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE_FROZEN, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


def cutmix_batch(images, masks, alpha=1.0, p=0.5):
    if random.random() > p:
        return images, masks
    B, C, H, W = images.shape
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = random.randint(0, H - cut_h)
    cx = random.randint(0, W - cut_w)
    indices = torch.randperm(B)
    images[:, :, cy:cy+cut_h, cx:cx+cut_w] = images[indices, :, cy:cy+cut_h, cx:cx+cut_w]
    masks[:, cy:cy+cut_h, cx:cx+cut_w] = masks[indices, cy:cy+cut_h, cx:cx+cut_w]
    return images, masks


# ---- CELL 5: LOAD DINOV2 ----
print("Loading DINOv2 ViT-B/14...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
backbone.to(device)
for param in backbone.parameters():
    param.requires_grad = False

with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    dummy_out = backbone.forward_features(dummy)["x_norm_patchtokens"]
    EMBED_DIM = dummy_out.shape[2]
del dummy, dummy_out
torch.cuda.empty_cache()

NUM_BLOCKS = len(backbone.blocks)
print(f"DINOv2: {EMBED_DIM}-dim, {NUM_BLOCKS} blocks")


# ---- CELL 6: SEGMENTATION HEAD ----
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
            nn.Dropout2d(0.15), nn.Conv2d(hidden, out_channels, 1),
        )
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.classifier(x)

seg_head = SegmentationHead(EMBED_DIM, NUM_CLASSES, TOKEN_W, TOKEN_H).to(device)
print(f"Head: {sum(p.numel() for p in seg_head.parameters()):,} params")


# ---- CELL 7: LOSS ----
class_freq = np.array([0.0353, 0.0593, 0.1887, 0.0110, 0.0439,
                        0.0281, 0.0008, 0.0120, 0.2445, 0.3764])
weights = 1.0 / np.sqrt(class_freq)
weights = weights / weights.min()
weights = np.clip(weights, 1.0, 15.0)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

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
        inter = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def compute_per_class_iou(preds, labels):
    preds = preds.argmax(1)
    ious = []
    for cls in range(NUM_CLASSES):
        inter = ((preds == cls) & (labels == cls)).sum().item()
        union = ((preds == cls) | (labels == cls)).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious


# ---- CELL 9: TRAINING ----
WARMUP_EPOCHS = 8
UNFREEZE_EPOCH = 9
TOTAL_EPOCHS = 80
UNFREEZE_BLOCKS = 6

optimizer = optim.AdamW(seg_head.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=100, eta_min=1e-5
)
scaler = GradScaler()
best_miou = 0.0

print(f"Phase 1: FROZEN x{WARMUP_EPOCHS} | Phase 2: UNFREEZE {UNFREEZE_BLOCKS}/{NUM_BLOCKS} blocks x{TOTAL_EPOCHS-WARMUP_EPOCHS}")
print(f"Augment: FDA + CutMix + ChannelShuffle + Grayscale + ExtremeColor")
print(f"{'='*60}\n")

for epoch in range(TOTAL_EPOCHS):
    if epoch + 1 == UNFREEZE_EPOCH:
        print(f"\n{'='*60}")
        print(f"UNFREEZING last {UNFREEZE_BLOCKS} blocks!")
        print(f"{'='*60}\n")
        for block in backbone.blocks[-UNFREEZE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(backbone, 'norm'):
            for param in backbone.norm.parameters():
                param.requires_grad = True
        optimizer = optim.AdamW([
            {"params": seg_head.parameters(), "lr": 5e-4},
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 3e-5},
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS - UNFREEZE_EPOCH + 1, eta_min=1e-6
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_UNFROZEN,
                                  shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_UNFROZEN,
                                shuffle=False, num_workers=2, pin_memory=True)
        gc.collect(); torch.cuda.empty_cache()

    seg_head.train()
    if epoch + 1 >= UNFREEZE_EPOCH:
        backbone.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        images, masks = cutmix_batch(images, masks, alpha=1.0, p=0.3)
        optimizer.zero_grad()

        if epoch + 1 < UNFREEZE_EPOCH:
            with torch.no_grad():
                features = backbone.forward_features(images)["x_norm_patchtokens"]
            with autocast():
                logits = seg_head(features)
                loss = hybrid_loss(logits, masks)
        else:
            with autocast():
                features = backbone.forward_features(images)["x_norm_patchtokens"]
                logits = seg_head(features)
                loss = hybrid_loss(logits, masks)

        scaler.scale(loss).backward()
        if epoch + 1 >= UNFREEZE_EPOCH:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(seg_head.parameters()) + [p for p in backbone.parameters() if p.requires_grad],
                max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    seg_head.eval(); backbone.eval()
    val_miou = 0.0
    all_class_ious = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            features = backbone.forward_features(images)["x_norm_patchtokens"]
            with autocast():
                logits = seg_head(features)
                logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            val_miou += compute_miou(logits.cpu(), masks.cpu())
            all_class_ious.append(compute_per_class_iou(logits.cpu(), masks.cpu()))

    val_miou /= len(val_loader)
    avg_class_ious = np.nanmean(all_class_ious, axis=0)
    phase = "FROZEN" if epoch + 1 <= WARMUP_EPOCHS else "FINETUNE"
    head_lr = optimizer.param_groups[0]["lr"]
    bb_lr = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else 0

    print(f"\n[{phase}] Epoch {epoch+1}/{TOTAL_EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val mIoU: {val_miou:.4f} | H-LR: {head_lr:.6f} | B-LR: {bb_lr:.6f}")

    if (epoch + 1) % 5 == 0 or val_miou > best_miou:
        print("  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = avg_class_ious[i]
            bar = "█" * int(v * 20) if not np.isnan(v) else ""
            val_str = f"{v:.3f}" if not np.isnan(v) else "N/A  "
            print(f"    {name:>16}: {val_str} {bar}")

    if val_miou > best_miou:
        best_miou = val_miou
        save_dict = {
            "seg_head": seg_head.state_dict(),
            "backbone_name": "dinov2_vitb14_reg",
            "embed_dim": EMBED_DIM, "token_w": TOKEN_W, "token_h": TOKEN_H,
            "num_classes": NUM_CLASSES, "img_size": [IMG_HEIGHT, IMG_WIDTH],
            "epoch": epoch + 1, "miou": val_miou,
        }
        if epoch + 1 >= UNFREEZE_EPOCH:
            save_dict["backbone_blocks"] = {}
            for i, block in enumerate(backbone.blocks[-UNFREEZE_BLOCKS:]):
                block_idx = NUM_BLOCKS - UNFREEZE_BLOCKS + i
                save_dict["backbone_blocks"][f"block_{block_idx}"] = block.state_dict()
            if hasattr(backbone, 'norm'):
                save_dict["backbone_norm"] = backbone.norm.state_dict()
        torch.save(save_dict, "best_model_v6.pth")
        print(f"  ★ NEW BEST: {best_miou:.4f}")

print(f"\n{'='*60}")
print(f"DONE — Best mIoU: {best_miou:.4f}")
print(f"{'='*60}")


# ---- CELL 10: SAVE CONFIG ----
config = {
    "model_name": "dinov2_vitb14_v6_domainrobust",
    "architecture": "DINOv2_ConvNeXtHead",
    "backbone": "dinov2_vitb14_reg",
    "embed_dim": EMBED_DIM, "num_classes": NUM_CLASSES,
    "input_size": [IMG_HEIGHT, IMG_WIDTH], "patch_size": PATCH_SIZE,
    "token_grid": [TOKEN_H, TOKEN_W], "unfrozen_blocks": UNFREEZE_BLOCKS,
    "class_names": CLASS_NAMES,
    "mask_value_mapping": {str(k): v for k, v in CLASS_TO_INDEX.items()},
    "preprocessing": {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "target_size": [IMG_HEIGHT, IMG_WIDTH],
    },
    "training": {
        "total_epochs": TOTAL_EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
        "unfreeze_epoch": UNFREEZE_EPOCH, "unfrozen_blocks": UNFREEZE_BLOCKS,
        "loss": "CE(label_smooth=0.05) + Dice",
        "augmentations": "FDA+CutMix+ChannelShuffle+Grayscale+ExtremeColor+Elastic",
        "best_miou": float(best_miou),
    },
}
with open("config_v6.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print("Saved: config_v6.yaml + best_model_v6.pth")
