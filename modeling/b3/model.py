import torch
import torch.nn as nn
import torchvision.models as models

class Baseline3_stg1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(self.backbone.fc.in_features, cfg.num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

class Baseline3_stg2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Pool over players
        self.pool = nn.AdaptiveMaxPool1d(1)

        # FC classifier
        self.scene_fc = nn.Sequential(
            nn.Linear(cfg.input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(1024, cfg.num_classes)
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=2048) — pre-extracted features
        batch, seq, players, feat = x.shape

        # 1. Pool over players per frame
        x = x.view(batch * seq, players, feat)       # (B*9, 12, 2048)
        x = x.permute(0, 2, 1)                       # (B*9, 2048, 12)
        x = self.pool(x).squeeze(-1)                  # (B*9, 2048)
        x = x.view(batch, seq, feat)                  # (B, 9, 2048)

        # 2. Pool over time (mean)
        x = x.mean(dim=1)                             # (B, 2048)

        # 3. Classify
        return self.scene_fc(x)                       # (B, num_classes)