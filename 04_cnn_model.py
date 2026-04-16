"""
04_cnn_model.py — CNN architecture for landscape image classification.
4 conv blocks (32->64->128->256) + GAP + 2 dense layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN_CHANNELS, CNN_DENSE_DIMS, DROPOUT_RATE, N_CLASSES


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class LandscapeCNN(nn.Module):
    """CNN for BBOB landscape image classification."""
    def __init__(self, n_classes=N_CLASSES, channels=CNN_CHANNELS,
                 dense_dims=CNN_DENSE_DIMS, dropout_rate=DROPOUT_RATE,
                 in_channels=3):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in channels:
            self.conv_blocks.append(ConvBlock(prev_ch, out_ch))
            prev_ch = out_ch

        self.last_channels = channels[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        layers = []
        in_dim = self.last_channels
        for out_dim in dense_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = out_dim

        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return self.classifier(x)

    def get_penultimate_embedding(self, x):
        """Return embedding from last dense layer (before classifier)."""
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(n_classes=N_CLASSES, in_channels=3):
    return LandscapeCNN(n_classes=n_classes, in_channels=in_channels)


if __name__ == "__main__":
    model = build_model()
    print("Model architecture:")
    print(model)
    print(f"\nTrainable params: {count_parameters(model):,}")

    batch = torch.randn(8, 3, 128, 128)
    logits = model(batch)
    print(f"Input: {batch.shape} -> Output: {logits.shape}")

    emb = model.get_penultimate_embedding(batch)
    print(f"Embedding: {emb.shape}")
