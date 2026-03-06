import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class Exp3_ResNet34(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet34(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Linear(base.fc.in_features, out_dim)
        self.model = base

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, p=2, dim=1)
