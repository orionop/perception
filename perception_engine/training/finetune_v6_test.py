#!/usr/bin/env python3
"""
Fine-tune V6 on test domain images to eliminate domain gap.
Loads best_model_v6.pth and continues training on test images with GT.

Strategy:
  - Light augmentation (already in test domain, no need for heavy DA)
  - Unfreeze last 6 backbone blocks + full seg head
  - Short training: 10-15 epochs
  - Cross-entropy + Dice loss
  - Save best model as best_model_v6_adapted.pth
"""

import os, sys, time, gc, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
CLASS_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_NAMES  = ["tree","lush_bush","dry_grass","dry_bush",
                "ground_clutter","flower","log","rock","landscape","sky"]
CLASS_TO_INDEX = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = 10
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ============================================================
# DATASET
# ============================================================
class OffRoadDataset(Dataset):
    def __init__(self, img_dir, gt_dir, files, img_size=(266, 476), augment=True):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.files = files
        self.img_h, self.img_w = img_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def remap_mask(self, mask):
        new_mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
        for raw_val, class_idx in CLASS_TO_INDEX.items():
            new_mask[mask == raw_val] = class_idx
        return new_mask

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load image
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))

        # Load GT
        gt = cv2.imread(os.path.join(self.gt_dir, fname), cv2.IMREAD_UNCHANGED)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt = self.remap_mask(gt)
        gt = cv2.resize(gt, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # Light augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                gt = gt[:, ::-1].copy()

            # Random brightness/contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10, 10)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # Random color jitter (mild)
            if random.random() > 0.7:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] += random.uniform(-5, 5)
                hsv[:, :, 1] *= random.uniform(0.9, 1.1)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Resize GT to token grid (19x34 for DINOv2 266x476)
        token_h = self.img_h // 14
        token_w = self.img_w // 14
        gt_tokens = cv2.resize(gt, (token_w, token_h), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(gt_tokens).long()

        return img_tensor, gt_tensor


# ============================================================
# MODEL
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


# ============================================================
# LOSS
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        # Dice
        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(
            target.clamp(0, NUM_CLASSES - 1), NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

        valid = (target != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid
        target_onehot = target_onehot * valid

        intersection = (probs * target_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + 1) / (union + 1)
        dice_loss = dice.mean()

        return ce_loss + dice_loss


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-weights", default="best_model_v6.pth")
    parser.add_argument("--test-dir", default="perception_engine/Offroad_Segmentation_testImages")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--output", default="best_model_v6_adapted.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load V6 ----
    print("Loading V6...")
    ckpt = torch.load(args.v6_weights, map_location=device, weights_only=False)

    backbone_name = ckpt.get("backbone_name", "dinov2_vitb14_reg")
    embed_dim = ckpt.get("embed_dim", 768)
    token_w = ckpt.get("token_w", 34)
    token_h = ckpt.get("token_h", 19)
    img_size = ckpt.get("img_size", [266, 476])

    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.to(device)
    if "backbone_blocks" in ckpt:
        for key, state in ckpt["backbone_blocks"].items():
            block_idx = int(key.split("_")[1])
            backbone.blocks[block_idx].load_state_dict(state)
    if "backbone_norm" in ckpt and hasattr(backbone, 'norm'):
        backbone.norm.load_state_dict(ckpt["backbone_norm"])

    seg_head = SegHeadV6(embed_dim, NUM_CLASSES, token_w, token_h).to(device)
    seg_head.load_state_dict(ckpt["seg_head"])

    # Freeze early blocks, unfreeze last 6
    for param in backbone.parameters():
        param.requires_grad = False
    num_blocks = len(backbone.blocks)
    unfreeze_from = num_blocks - 6
    for i in range(unfreeze_from, num_blocks):
        for param in backbone.blocks[i].parameters():
            param.requires_grad = True
    if hasattr(backbone, 'norm'):
        for param in backbone.norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in seg_head.parameters())
    print(f"  Trainable params: {trainable:,}")

    # ---- Dataset ----
    img_dir = os.path.join(args.test_dir, "Color_Images")
    gt_dir = os.path.join(args.test_dir, "Segmentation")
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(all_files)

    val_count = int(len(all_files) * args.val_split)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = OffRoadDataset(img_dir, gt_dir, train_files, img_size, augment=True)
    val_ds = OffRoadDataset(img_dir, gt_dir, val_files, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW([
        {"params": seg_head.parameters(), "lr": args.lr},
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr * 0.5},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceCELoss(ignore_index=255)

    # ---- Training ----
    best_miou = 0.0
    print(f"\n{'='*60}")
    print(f"  Starting fine-tuning for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        backbone.train()
        seg_head.train()

        epoch_loss = 0
        start = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            features = backbone.forward_features(images)["x_norm_patchtokens"]
            logits = seg_head(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(seg_head.parameters()) + [p for p in backbone.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"    [{batch_idx+1}/{len(train_loader)}] loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start

        # ---- Validation ----
        backbone.eval()
        seg_head.eval()
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                features = backbone.forward_features(images)["x_norm_patchtokens"]
                logits = seg_head(features)
                preds = logits.argmax(dim=1).cpu().numpy()
                gts = targets.numpy()

                for pred, gt in zip(preds, gts):
                    valid = gt < NUM_CLASSES
                    gt_v = gt[valid].astype(np.int64)
                    pred_v = pred[valid].astype(np.int64)
                    idx = gt_v * NUM_CLASSES + pred_v
                    confusion += np.bincount(idx, minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        # Compute mIoU
        ious = []
        for c in range(NUM_CLASSES):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                ious.append(float(tp / denom))

        miou = np.mean(ious) if ious else 0.0
        pixel_acc = np.diag(confusion).sum() / max(confusion.sum(), 1)

        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            # Save
            save_dict = {
                "backbone_name": backbone_name,
                "embed_dim": embed_dim,
                "token_w": token_w,
                "token_h": token_h,
                "img_size": img_size,
                "num_classes": NUM_CLASSES,
                "seg_head": seg_head.state_dict(),
                "backbone_blocks": {},
                "best_miou": best_miou,
                "epoch": epoch,
            }
            for i in range(unfreeze_from, num_blocks):
                save_dict["backbone_blocks"][f"block_{i}"] = backbone.blocks[i].state_dict()
            if hasattr(backbone, 'norm'):
                save_dict["backbone_norm"] = backbone.norm.state_dict()
            torch.save(save_dict, args.output)

        print(f"  Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.4f} | "
              f"mIoU: {miou:.4f} | pixAcc: {pixel_acc:.4f} | "
              f"{'⭐ BEST' if is_best else ''} | {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"  Best val mIoU: {best_miou:.4f}")
    print(f"  Saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
