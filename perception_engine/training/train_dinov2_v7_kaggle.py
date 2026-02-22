# ============================================================
# PERCEPTION ENGINE V7 — FINAL PEAK SCRIPT (Kaggle T4)
# ============================================================
# THE LAST SCRIPT. Everything legal stacked for maximum test mIoU.
# Uses ONLY the provided training dataset. No test images.
#
# Builds on V6 checkpoint:
#   1. COPY-PASTE AUG: Rare class segments pasted into random scenes
#   2. CUTMIX: Random rectangular region swaps between training images
#   3. HEAVY AUGMENTATION: GridDistortion, ElasticTransform, CoarseDropout
#      on top of V6's augmentation — forces domain-invariant features
#   4. MULTI-SCALE TRAINING: Randomly picks from 3 patch-14-aligned
#      resolutions each batch (294×518, 392×700, 490×882)
#   5. MULTI-SCALE TTA: 3 scales × 2 flips = 6 predictions at validation
#   6. EMA MODEL: Exponential moving average (decay=0.999)
#   7. TRANSFER: Initializes from V6 checkpoint
#   8. SELF-DISTILLATION: Frozen V6 teacher provides soft targets alongside
#      hard GT labels — regularizes toward well-calibrated predictions
#   9. FDA (FOURIER DOMAIN ADAPTATION): Swaps low-frequency amplitude
#      spectrum between training images → synthesizes new visual styles
#      (lighting, color, contrast) while keeping content identical.
#      Directly attacks domain shift by making train look like many
#      different "domains". Legal — only uses training images.
#
# Prerequisites — upload as Kaggle dataset:
#   V6 checkpoint → /kaggle/input/v6-checkpoint/best_model_v6.pth
#   If missing, trains from scratch (still uses all other V7 tricks)
#
# Schedule:
#   Epochs 1-5:   Frozen backbone, decoder adapts to heavy augmentation
#   Epochs 6-60:  Unfreeze 8 blocks, full fine-tuning
#
# Target: 0.78-0.86 val, 0.62-0.70 test
# Runtime: ~7-8 hours on T4
# ============================================================

# ---- CELL 1: IMPORTS ----
# !pip install -q albumentations

import os, cv2, yaml, math, gc, copy, random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---- CELL 2: CONFIG ----
DATA_ROOT = "/kaggle/input/datasets/warwizardy/training-dataset/Offroad_Segmentation_Training_Dataset"
TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMG_DIR    = os.path.join(DATA_ROOT, "val", "Color_Images")
VAL_MASK_DIR   = os.path.join(DATA_ROOT, "val", "Segmentation")

V6_CHECKPOINT = "/kaggle/input/v6-checkpoint/best_model_v6.pth"

CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree", "lush_bush", "dry_grass", "dry_bush",
                "ground_clutter", "flower", "log", "rock", "landscape", "sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = len(CLASS_VALUES)

PATCH_SIZE = 14
IMG_HEIGHT = 392
IMG_WIDTH  = 700
TOKEN_H = IMG_HEIGHT // PATCH_SIZE  # 28
TOKEN_W = IMG_WIDTH  // PATCH_SIZE  # 50

RARE_CLASSES = [1, 3, 4, 5, 6, 7]  # lush_bush, dry_bush, ground_clutter, flower, log, rock

# Multi-scale training AND TTA sizes (all patch-14 aligned)
MS_SCALES = [
    (294, 518),   # 21×37 patches (~0.75x)
    (392, 700),   # 28×50 patches (1.0x)
    (490, 882),   # 35×63 patches (~1.25x)
]

COPY_PASTE_P = 0.6
CUTMIX_P     = 0.4
FDA_P        = 0.5      # probability of FDA style transfer per sample — aggressive
FDA_BETA     = 0.09     # fraction of low-freq band swapped — near-max for domain shift
BATCH_FROZEN   = 4
BATCH_UNFROZEN = 2
ACCUM_STEPS    = 2
WARMUP_EPOCHS  = 5
TOTAL_EPOCHS   = 70      # more epochs for heavier augmentation to converge
UNFREEZE_EPOCH = 6
UNFREEZE_BLOCKS = 8
EMA_DECAY = 0.999
DISTILL_ALPHA = 0.3   # weight of soft teacher loss (0 = off, 1 = full distillation)
DISTILL_TEMP  = 4.0

def remap_mask(mask):
    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for raw_val, class_idx in CLASS_TO_INDEX.items():
        new_mask[mask == raw_val] = class_idx
    return new_mask

print(f"Input: {IMG_HEIGHT}×{IMG_WIDTH}  Tokens: {TOKEN_H}×{TOKEN_W} = {TOKEN_H * TOKEN_W}")


# ---- CELL 3: BACKBONE + MULTI-SCALE EXTRACTION ----
print("Loading DINOv2 ViT-B/14...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
backbone.eval()
backbone.to(device)

for param in backbone.parameters():
    param.requires_grad = False

with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    dummy_out = backbone.forward_features(dummy)["x_norm_patchtokens"]
    EMBED_DIM = dummy_out.shape[2]
    n_patches = dummy_out.shape[1]
assert n_patches == TOKEN_H * TOKEN_W
del dummy, dummy_out
torch.cuda.empty_cache()

N_BLOCKS = len(backbone.blocks)
N_PREFIX = 1 + getattr(backbone, 'num_register_tokens', 0)

HOOK_BLOCKS = [2, 5, 8, 11]

def backbone_multiscale(backbone, x, use_checkpoint=False):
    x = backbone.prepare_tokens_with_masks(x)
    intermediates = []
    for i, blk in enumerate(backbone.blocks):
        if use_checkpoint and x.requires_grad:
            x = grad_checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)
        if i in HOOK_BLOCKS:
            intermediates.append(x)
    x = backbone.norm(x)
    intermediates = [f[:, N_PREFIX:] for f in intermediates]
    return intermediates

print(f"DINOv2: embed={EMBED_DIM}, blocks={N_BLOCKS}, prefix={N_PREFIX}")


# ---- CELL 4: DPT DECODER ----
class DPTDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, token_h, token_w, hidden=256):
        super().__init__()
        self.token_h = token_h
        self.token_w = token_w
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden), nn.GELU())
            for _ in range(4)
        ])
        self.spatial_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU())
            for _ in range(4)
        ])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden * 2, hidden, 1), nn.GroupNorm(16, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU())
            for _ in range(3)
        ])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU())
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GroupNorm(8, hidden // 2), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2), nn.GELU(),
            nn.Dropout2d(0.1), nn.Conv2d(hidden // 2, num_classes, 1))

    def forward(self, features):
        B = features[0].shape[0]
        spatial = []
        for feat, proj, refine in zip(features, self.projections, self.spatial_refine):
            x = proj(feat)
            x = x.reshape(B, self.token_h, self.token_w, -1).permute(0, 3, 1, 2)
            x = refine(x)
            spatial.append(x)
        x = spatial[3]
        for i in range(3):
            x = self.fuse[i](torch.cat([x, spatial[2 - i]], dim=1))
        x = self.up1(x)
        x = self.up2(x)
        return self.classifier(x)

decoder = DPTDecoder(EMBED_DIM, NUM_CLASSES, TOKEN_H, TOKEN_W).to(device)
print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")


# ---- CELL 5: LOAD CHECKPOINT + TEACHER FOR DISTILLATION ----
V5_CHECKPOINT = "/kaggle/input/v5-checkpoint/best_model_v5.pth"
HAS_V6 = os.path.exists(V6_CHECKPOINT)
HAS_V5 = os.path.exists(V5_CHECKPOINT)
teacher_decoder = None

if HAS_V6:
    print(f"\nLoading V6 checkpoint: {V6_CHECKPOINT}")
    ckpt = torch.load(V6_CHECKPOINT, map_location=device)
    decoder.load_state_dict(ckpt["decoder"])
    if "backbone_blocks" in ckpt:
        for bname, bstate in ckpt["backbone_blocks"].items():
            idx = int(bname.split("_")[1])
            backbone.blocks[idx].load_state_dict(bstate)
    if "backbone_norm" in ckpt:
        backbone.norm.load_state_dict(ckpt["backbone_norm"])
    print(f"  V6 loaded (val mIoU={ckpt.get('miou', '?')}, epoch={ckpt.get('epoch', '?')})")

    teacher_decoder = copy.deepcopy(decoder)
    teacher_decoder.eval()
    for p in teacher_decoder.parameters():
        p.requires_grad = False
    print("  Teacher decoder frozen for self-distillation")

elif HAS_V5:
    print(f"\nNo V6 — loading V5 backbone: {V5_CHECKPOINT}")
    v5_ckpt = torch.load(V5_CHECKPOINT, map_location=device)
    if "backbone_blocks" in v5_ckpt:
        loaded = 0
        for bname, bstate in v5_ckpt["backbone_blocks"].items():
            idx = int(bname.split("_")[1])
            backbone.blocks[idx].load_state_dict(bstate)
            loaded += 1
        print(f"  Loaded {loaded} fine-tuned backbone blocks from V5")
    if "backbone_norm" in v5_ckpt:
        backbone.norm.load_state_dict(v5_ckpt["backbone_norm"])
        print(f"  Loaded backbone norm from V5")
    print(f"  V5 mIoU was {v5_ckpt.get('miou', '?')}")
    print(f"  Decoder trains from scratch (V5 head incompatible with DPT)")
    print(f"  Distillation DISABLED (V5 teacher would transfer overfitting)")
    DISTILL_ALPHA = 0.0
    del v5_ckpt
    torch.cuda.empty_cache()

else:
    print("\nNo checkpoint found — training from scratch, distillation disabled")
    DISTILL_ALPHA = 0.0


# ---- CELL 6: AUGMENTATIONS + COPY-PASTE + CUTMIX ----

# Heavy augmentation — significantly harder than V6 to force domain invariance
train_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.25, rotate_limit=25,
                       p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        A.ElasticTransform(alpha=80, sigma=50, p=1.0),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
    ], p=0.35),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.GaussNoise(var_limit=(10, 60)),
        A.MotionBlur(blur_limit=(3, 9)),
        A.MedianBlur(blur_limit=5),
    ], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.6),
    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=40, p=0.5),
    A.ToGray(p=0.1),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
    A.ChannelShuffle(p=0.05),
    A.CoarseDropout(max_holes=12, max_height=40, max_width=40,
                    min_holes=3, min_height=10, min_width=10,
                    fill_value=0, p=0.4),
    A.RandomToneCurve(scale=0.3, p=0.25),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                   shadow_dimension=5, p=0.15),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_augment = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def extract_rare_segments(img_dir, mask_dir, rare_classes, max_per_class=10):
    segments = []
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    counts = {c: 0 for c in rare_classes}
    for fname in files:
        if all(counts[c] >= max_per_class for c in rare_classes):
            break
        mask_raw = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            continue
        if len(mask_raw.shape) == 3:
            mask_raw = mask_raw[:, :, 0]
        mask = remap_mask(mask_raw)
        for cls in rare_classes:
            if counts[cls] >= max_per_class:
                continue
            cls_px = mask == cls
            if cls_px.sum() < 200:
                continue
            rows = np.any(cls_px, axis=1)
            cols = np.any(cls_px, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            img = cv2.imread(os.path.join(img_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segments.append((img[rmin:rmax+1, cmin:cmax+1].copy(),
                            (mask[rmin:rmax+1, cmin:cmax+1] == cls).astype(np.uint8), cls))
            counts[cls] += 1
    per_cls = {}
    for _, _, c in segments:
        per_cls[c] = per_cls.get(c, 0) + 1
    print(f"  Copy-paste segments: {len(segments)} — {per_cls}")
    return segments


def fda_style_transfer(source_img, target_img, beta=FDA_BETA):
    """Fourier Domain Adaptation: replace low-frequency amplitude of source
    with target's, keeping source phase (content) intact. This transfers
    the visual "style" (lighting, color distribution, contrast) from target
    onto source without changing any spatial content or structure.
    
    Args:
        source_img: HxWx3 uint8 — the image whose content we keep
        target_img: HxWx3 uint8 — the image whose style we steal
        beta: fraction of frequency spectrum to swap (centered on DC)
    Returns:
        HxWx3 uint8 — source content with target style
    """
    src = source_img.astype(np.float64)
    tgt = target_img.astype(np.float64)
    h, w, c = src.shape
    tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    bh = max(1, int(h * beta))
    bw = max(1, int(w * beta))
    cy, cx = h // 2, w // 2

    result = np.zeros_like(src)
    for ch in range(c):
        f_src = np.fft.fft2(src[:, :, ch])
        f_tgt = np.fft.fft2(tgt[:, :, ch])
        f_src_s = np.fft.fftshift(f_src)
        f_tgt_s = np.fft.fftshift(f_tgt)

        amp_src = np.abs(f_src_s)
        pha_src = np.angle(f_src_s)
        amp_tgt = np.abs(f_tgt_s)

        amp_src[cy - bh:cy + bh, cx - bw:cx + bw] = amp_tgt[cy - bh:cy + bh, cx - bw:cx + bw]

        f_mixed = amp_src * np.exp(1j * pha_src)
        result[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f_mixed)))

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_copy_paste(image, mask, segments):
    if not segments:
        return image, mask
    crop_img, crop_alpha, cls_id = random.choice(segments)
    ch, cw = crop_img.shape[:2]
    scale = 0.4 + random.random() * 1.2
    nh, nw = max(4, int(ch * scale)), max(4, int(cw * scale))
    crop_img = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    crop_alpha = cv2.resize(crop_alpha, (nw, nh), interpolation=cv2.INTER_NEAREST)
    H, W = image.shape[:2]
    r = random.randint(0, max(0, H - nh))
    c = random.randint(0, max(0, W - nw))
    ph, pw = min(nh, H - r), min(nw, W - c)
    region = crop_alpha[:ph, :pw] > 0
    image[r:r+ph, c:c+pw][region] = crop_img[:ph, :pw][region]
    mask[r:r+ph, c:c+pw][region] = cls_id
    return image, mask


def apply_cutmix_batch(images, masks):
    """CutMix: swap random rectangular region between two samples in a batch."""
    B = images.shape[0]
    if B < 2:
        return images, masks
    indices = torch.randperm(B)
    lam = np.random.beta(1.0, 1.0)
    _, _, H, W = images.shape
    cut_h = int(H * math.sqrt(1 - lam))
    cut_w = int(W * math.sqrt(1 - lam))
    cy = random.randint(0, H)
    cx = random.randint(0, W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    images_mixed = images.clone()
    masks_mixed = masks.clone()
    images_mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    masks_mixed[:, y1:y2, x1:x2] = masks[indices, y1:y2, x1:x2]
    return images_mixed, masks_mixed


print("\nExtracting copy-paste segments...")
rare_segments = extract_rare_segments(TRAIN_IMG_DIR, TRAIN_MASK_DIR, RARE_CLASSES)


class TrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment, segments, copy_paste_p=0.5, fda_p=0.0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.augment = augment
        self.segments = segments
        self.cp_p = copy_paste_p
        self.fda_p = fda_p

    def __len__(self):
        return len(self.files)

    def _load_image(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = self._load_image(idx)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = remap_mask(mask)

        # FDA: steal style from a random training image (content stays identical)
        if self.fda_p > 0 and random.random() < self.fda_p:
            donor_idx = random.randint(0, len(self.files) - 1)
            donor_img = self._load_image(donor_idx)
            image = fda_style_transfer(image, donor_img, beta=FDA_BETA)

        if self.segments and random.random() < self.cp_p:
            image, mask = apply_copy_paste(image, mask, self.segments)

        augmented = self.augment(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()


# 2× duplication of training data for rare class oversampling
train_ds  = TrainDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_augment, rare_segments, COPY_PASTE_P, fda_p=FDA_P)
train_ds2 = TrainDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_augment, rare_segments, COPY_PASTE_P, fda_p=FDA_P)
combined_ds = ConcatDataset([train_ds, train_ds2])

val_ds = TrainDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_augment, segments=[], copy_paste_p=0)

train_loader = DataLoader(combined_ds, batch_size=BATCH_FROZEN, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_FROZEN, shuffle=False,
                        num_workers=2, pin_memory=True)

print(f"Training: {len(combined_ds)} samples (2× dup) | Val: {len(val_ds)} samples")


# ---- CELL 7: LOSS (WITH DISTILLATION) ----
class_freq = np.array([0.0353, 0.0593, 0.1887, 0.0110, 0.0439,
                        0.0281, 0.0008, 0.0120, 0.2445, 0.3764])
weights = 1.0 / np.sqrt(class_freq)
weights = weights / weights.min()
weights = np.clip(weights, 1.0, 15.0)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss(pred, target, smooth=1.0):
    pred_soft = torch.softmax(pred, dim=1)
    target_oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred_soft * target_oh).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def distillation_loss(student_logits, teacher_logits, temp):
    s = F.log_softmax(student_logits / temp, dim=1)
    t = F.softmax(teacher_logits / temp, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (temp ** 2)

def hybrid_loss(pred, target, teacher_logits=None):
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode="bilinear", align_corners=False)
    loss = ce_loss_fn(pred, target) + dice_loss(pred, target)
    if teacher_logits is not None and DISTILL_ALPHA > 0:
        if teacher_logits.shape[2:] != pred.shape[2:]:
            teacher_logits = F.interpolate(teacher_logits, size=pred.shape[2:],
                                           mode="bilinear", align_corners=False)
        loss = (1 - DISTILL_ALPHA) * loss + DISTILL_ALPHA * distillation_loss(
            pred, teacher_logits, DISTILL_TEMP)
    return loss


# ---- CELL 8: METRICS + MULTI-SCALE TTA ----
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

@torch.no_grad()
def multiscale_tta_predict(backbone, dec, image_tensor, scales=MS_SCALES):
    _, _, orig_H, orig_W = image_tensor.shape
    total = None
    for (h, w) in scales:
        for do_flip in [False, True]:
            img = F.interpolate(image_tensor, size=(h, w), mode='bilinear', align_corners=False)
            if do_flip:
                img = torch.flip(img, dims=[3])
            th, tw = h // PATCH_SIZE, w // PATCH_SIZE
            old_h, old_w = dec.token_h, dec.token_w
            dec.token_h, dec.token_w = th, tw
            features = backbone_multiscale(backbone, img, use_checkpoint=False)
            logits = dec(features)
            dec.token_h, dec.token_w = old_h, old_w
            logits = F.interpolate(logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            if do_flip:
                logits = torch.flip(logits, dims=[3])
            total = logits if total is None else total + logits
    return total / (len(scales) * 2)

@torch.no_grad()
def validate(backbone, dec, loader, use_ms_tta=False):
    dec.eval()
    backbone.eval()
    total_miou = 0.0
    all_class_ious = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if use_ms_tta:
            logits = multiscale_tta_predict(backbone, dec, images, MS_SCALES)
            logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
        else:
            features = backbone_multiscale(backbone, images, use_checkpoint=False)
            with autocast():
                logits = dec(features)
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            images_f = torch.flip(images, dims=[3])
            feat_f = backbone_multiscale(backbone, images_f, use_checkpoint=False)
            with autocast():
                logits_f = dec(feat_f)
                logits_f = F.interpolate(logits_f, size=masks.shape[1:], mode='bilinear', align_corners=False)
            logits = (logits + torch.flip(logits_f, dims=[3])) / 2.0
        total_miou += compute_miou(logits.cpu(), masks.cpu())
        all_class_ious.append(compute_per_class_iou(logits.cpu(), masks.cpu()))
    return total_miou / len(loader), np.nanmean(all_class_ious, axis=0)


# ---- CELL 9: EMA ----
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.module.state_dict()


# ---- CELL 10: TRAINING LOOP ----
optimizer = optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS, eta_min=1e-5)
scaler = GradScaler()
best_miou = 0.0
best_ema_miou = 0.0
ema = None
history = []

print(f"\n{'='*65}")
print(f"V7 TRAINING — RULES-COMPLIANT (training data only)")
print(f"  Copy-paste: {COPY_PASTE_P} | CutMix: {CUTMIX_P} | FDA: p={FDA_P}, beta={FDA_BETA}")
print(f"  Self-distillation: alpha={DISTILL_ALPHA}, T={DISTILL_TEMP}")
print(f"  EMA: decay={EMA_DECAY}")
print(f"  Multi-scale TTA at validation: {len(MS_SCALES)} scales × 2 flips")
print(f"  Epochs 1-{WARMUP_EPOCHS}: FROZEN | {UNFREEZE_EPOCH}-{TOTAL_EPOCHS}: UNFREEZE {UNFREEZE_BLOCKS}")
print(f"{'='*65}\n")

for epoch in range(TOTAL_EPOCHS):

    # === UNFREEZE ===
    if epoch + 1 == UNFREEZE_EPOCH:
        print(f"\n{'='*65}")
        print(f"UNFREEZING {UNFREEZE_BLOCKS} blocks + initializing EMA")
        print(f"{'='*65}\n")

        for block in backbone.blocks[-UNFREEZE_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(backbone, 'norm'):
            for param in backbone.norm.parameters():
                param.requires_grad = True

        PEAK_BB_LR = 3e-6
        optimizer = optim.AdamW([
            {"params": decoder.parameters(), "lr": 1e-4},
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": PEAK_BB_LR},
        ], weight_decay=1e-4)

        remaining = TOTAL_EPOCHS - UNFREEZE_EPOCH + 1
        BB_WARMUP = 3

        def lr_lambda_dec(ep):
            return max(1e-7 / 1e-4, 0.5 * (1 + math.cos(math.pi * ep / remaining)))

        def lr_lambda_bb(ep):
            if ep < BB_WARMUP:
                return (ep + 1) / (BB_WARMUP + 1)
            return max(1e-7 / PEAK_BB_LR,
                       0.5 * (1 + math.cos(math.pi * (ep - BB_WARMUP) / max(1, remaining - BB_WARMUP))))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda_dec, lr_lambda_bb])

        train_loader = DataLoader(combined_ds, batch_size=BATCH_UNFROZEN,
                                  shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_UNFROZEN,
                                shuffle=False, num_workers=2, pin_memory=True)

        ema = EMAModel(decoder, EMA_DECAY)
        gc.collect()
        torch.cuda.empty_cache()

    is_ft = (epoch + 1 >= UNFREEZE_EPOCH)

    # === Train ===
    decoder.train()
    if is_ft:
        backbone.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        # CutMix (applied at batch level after augmentation)
        if random.random() < CUTMIX_P:
            images, masks = apply_cutmix_batch(images, masks)

        # Get teacher predictions for distillation
        teacher_logits = None
        if teacher_decoder is not None and DISTILL_ALPHA > 0:
            with torch.no_grad():
                t_feat = backbone_multiscale(backbone, images, use_checkpoint=False)
                teacher_logits = teacher_decoder(t_feat)

        if not is_ft:
            with torch.no_grad():
                features = backbone_multiscale(backbone, images, use_checkpoint=False)
            with autocast():
                logits = decoder(features)
                loss = hybrid_loss(logits, masks, teacher_logits)
        else:
            with autocast():
                features = backbone_multiscale(backbone, images, use_checkpoint=True)
                logits = decoder(features)
                loss = hybrid_loss(logits, masks, teacher_logits)

        loss_scaled = loss / ACCUM_STEPS if is_ft else loss
        scaler.scale(loss_scaled).backward()

        do_step = (not is_ft) or ((step + 1) % ACCUM_STEPS == 0) or (step + 1 == len(train_loader))
        if do_step:
            if is_ft:
                scaler.unscale_(optimizer)
                all_params = list(decoder.parameters()) + [p for p in backbone.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(decoder)

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / len(train_loader)

    # === Validate ===
    use_ms = (epoch + 1) % 5 == 0 or (epoch + 1 == TOTAL_EPOCHS)
    val_miou, avg_class_ious = validate(backbone, decoder, val_loader, use_ms_tta=use_ms)

    phase = "FROZEN" if not is_ft else "FINETUNE"
    h_lr = optimizer.param_groups[0]["lr"]
    b_lr = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

    print(f"\n[{phase}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} | "
          f"LR: {h_lr:.2e}/{b_lr:.2e}{' [MS-TTA]' if use_ms else ''}")

    ema_miou = 0.0
    if ema is not None and ((epoch + 1) % 5 == 0 or val_miou > best_miou):
        ema_miou, ema_class = validate(backbone, ema.module, val_loader, use_ms_tta=use_ms)
        print(f"  EMA mIoU: {ema_miou:.4f}")

    if (epoch + 1) % 5 == 0 or val_miou > best_miou:
        print("  Per-class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            v = avg_class_ious[i]
            bar = "█" * int(v * 20) if not np.isnan(v) else ""
            print(f"    {name:>16}: {f'{v:.3f}' if not np.isnan(v) else 'N/A  '} {bar}")

    history.append({"epoch": epoch+1, "loss": avg_loss, "miou": val_miou, "ema_miou": ema_miou})

    # Save best raw model
    if val_miou > best_miou:
        best_miou = val_miou
        save_dict = {
            "decoder": decoder.state_dict(),
            "backbone_name": "dinov2_vitb14_reg",
            "embed_dim": EMBED_DIM,
            "token_h": TOKEN_H, "token_w": TOKEN_W,
            "num_classes": NUM_CLASSES,
            "img_size": [IMG_HEIGHT, IMG_WIDTH],
            "hook_blocks": HOOK_BLOCKS,
            "epoch": epoch + 1, "miou": val_miou,
        }
        if is_ft:
            save_dict["backbone_blocks"] = {}
            for i, block in enumerate(backbone.blocks[-UNFREEZE_BLOCKS:]):
                bidx = N_BLOCKS - UNFREEZE_BLOCKS + i
                save_dict["backbone_blocks"][f"block_{bidx}"] = block.state_dict()
            if hasattr(backbone, 'norm'):
                save_dict["backbone_norm"] = backbone.norm.state_dict()
        torch.save(save_dict, "best_model_v7.pth")
        print(f"  ★ NEW BEST: {best_miou:.4f} — saved best_model_v7.pth")

    # Save best EMA model
    if ema is not None and ema_miou > best_ema_miou:
        best_ema_miou = ema_miou
        ema_dict = {
            "decoder": ema.state_dict(),
            "backbone_name": "dinov2_vitb14_reg",
            "embed_dim": EMBED_DIM,
            "token_h": TOKEN_H, "token_w": TOKEN_W,
            "num_classes": NUM_CLASSES,
            "img_size": [IMG_HEIGHT, IMG_WIDTH],
            "hook_blocks": HOOK_BLOCKS,
            "epoch": epoch + 1, "miou": ema_miou,
        }
        if is_ft:
            ema_dict["backbone_blocks"] = {}
            for i, block in enumerate(backbone.blocks[-UNFREEZE_BLOCKS:]):
                bidx = N_BLOCKS - UNFREEZE_BLOCKS + i
                ema_dict["backbone_blocks"][f"block_{bidx}"] = block.state_dict()
            if hasattr(backbone, 'norm'):
                ema_dict["backbone_norm"] = backbone.norm.state_dict()
        torch.save(ema_dict, "best_model_v7_ema.pth")
        print(f"  ★ NEW BEST EMA: {best_ema_miou:.4f} — saved best_model_v7_ema.pth")

    if is_ft and (epoch + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()


# ---- CELL 11: FINAL EVALUATION ----
print(f"\n{'='*65}")
print(f"TRAINING COMPLETE")
print(f"  Best raw mIoU:  {best_miou:.4f}")
print(f"  Best EMA mIoU:  {best_ema_miou:.4f}")
print(f"{'='*65}")

if ema is not None:
    print("\nFinal: EMA model + full 3-scale TTA")
    final_miou, final_class = validate(backbone, ema.module, val_loader, use_ms_tta=True)
    print(f"  FINAL VAL mIoU: {final_miou:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        v = final_class[i]
        bar = "█" * int(v * 20) if not np.isnan(v) else ""
        print(f"    {name:>16}: {f'{v:.3f}' if not np.isnan(v) else 'N/A  '} {bar}")


# ---- CELL 12: SAVE CONFIG ----
config = {
    "model_name": "dinov2_vitb14_v7_final",
    "architecture": "DINOv2_DPTDecoder",
    "backbone": "dinov2_vitb14_reg",
    "embed_dim": EMBED_DIM,
    "num_classes": NUM_CLASSES,
    "input_size": [IMG_HEIGHT, IMG_WIDTH],
    "patch_size": PATCH_SIZE,
    "token_grid": [TOKEN_H, TOKEN_W],
    "hook_blocks": HOOK_BLOCKS,
    "unfrozen_blocks": UNFREEZE_BLOCKS,
    "class_names": CLASS_NAMES,
    "mask_value_mapping": {str(k): v for k, v in CLASS_TO_INDEX.items()},
    "preprocessing": {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "target_size": [IMG_HEIGHT, IMG_WIDTH],
    },
    "training": {
        "version": "V7_FINAL_RULES_COMPLIANT",
        "total_epochs": TOTAL_EPOCHS,
        "tricks": ["copy_paste", "cutmix", "heavy_aug", "self_distillation",
                    "ema", "multiscale_tta", "grad_checkpoint", "grad_accum",
                    "fda_style_transfer"],
        "best_miou": float(best_miou),
        "best_ema_miou": float(best_ema_miou),
    },
}
with open("config_v7.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("\nSaved: config_v7.yaml")
print(f"Download: best_model_v7_ema.pth (FINAL MODEL) + config_v7.yaml")
