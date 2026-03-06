import torch.nn as nn
import torch.nn.functional as F

class Exp2_CNN6(nn.Module):
    def __init__(self, out_dim=128, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 6 * 6, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
