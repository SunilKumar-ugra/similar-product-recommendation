# src/modeling/model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
