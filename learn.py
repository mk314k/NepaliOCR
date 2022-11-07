"""
PyTorch CRNN + CTC sequence recognition model for Devanagari and English word crops.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility import CTC_BLANK_INDEX, encode_text, unicodes


class _ResidualConvBlock(nn.Module):
    """Two-layer `3x3` residual block with batch normalization (`momentum=0.2`)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.2)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.2),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class _DevanagariCRNNNet(nn.Module):
    """
    Residual CNN + BiLSTM network mapping `(B, 1, 32, W)` to `(B, W // 4, num_classes)`.

    Unlike Latin CRNNs that pool height all the way to `H = 1`, this backbone stops
    vertical pooling at `H = 4` (`32 -> 16 -> 8 -> 4`) and flattens `C * H = 512`
    features per horizontal step before a linear projection to `256`. Keeping 4
    vertical bands preserves spatial separation between upper matras (`े`, `ै`, `ं`),
    the middle consonant body, and lower subscript matras (`ु`, `ू`, `ृ`, `्`).
    """

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        # (B, 1, 32, W) -> (B, 64, 16, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # (B, 64, 16, W/2) -> (B, 128, 8, W/4)
        self.stage2 = nn.Sequential(
            _ResidualConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # (B, 128, 8, W/4) -> (B, 128, 4, W/4)
        self.stage3 = nn.Sequential(
            _ResidualConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _ResidualConvBlock(128, 128),
        )
        self.vertical_proj = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.blstm1 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(512)
        self.blstm2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)

        # Initialize blank logit bias slightly negative so early CTC steps do not
        # collapse to all-blank predictions.
        nn.init.constant_(self.fc.bias, 0.0)
        with torch.no_grad():
            self.fc.bias[num_classes - 1] = -2.0

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous() or x.stride(1) != x.size(2) * x.size(3):
            x = x.clone(memory_format=torch.contiguous_format)
        feat = self.stem(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        B, C, H, T = feat.shape
        seq = (
            feat.permute(0, 3, 1, 2)
            .clone(memory_format=torch.contiguous_format)
            .view(B, T, C * H)
        )
        seq = self.vertical_proj(seq)
        out1, _ = self.blstm1(seq)
        out1 = self.ln1(out1)
        out2, _ = self.blstm2(out1)
        out2 = self.ln2(out2)
        return self.fc(out2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(x), dim=-1)


