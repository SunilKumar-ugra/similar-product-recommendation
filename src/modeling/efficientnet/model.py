# src/modeling/efficientnet/model.py
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)   # (B, 1280)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
