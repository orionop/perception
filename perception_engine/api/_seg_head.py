"""Segmentation head matching the V6 checkpoint format exactly."""

import torch.nn as nn


class SegmentationHead(nn.Module):
    """V6 head — 3 ConvNeXt blocks.
    Matches the saved checkpoint keys: stem, block1, block2, block3, classifier.
    Input is spatial (B, C, H, W).
    """
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        h = 256
        self.stem = nn.Sequential(nn.Conv2d(in_ch, h, 7, padding=3), nn.BatchNorm2d(h), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(h, h, 7, padding=3, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(h, h, 5, padding=2, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.block3 = nn.Sequential(nn.Conv2d(h, h, 3, padding=1, groups=h), nn.BatchNorm2d(h), nn.GELU(), nn.Conv2d(h, h, 1), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout2d(0.15), nn.Conv2d(h, out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.classifier(x)
