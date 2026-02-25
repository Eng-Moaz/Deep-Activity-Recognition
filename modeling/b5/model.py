import torch
import torch.nn as nn
import torchvision.models as models


class Baseline5_stg1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(cfg.hidden_size)  # stable scale for stage-2 features

        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, Seq=9, C, H, W) — one player's crops over time
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)             # (B*9, 3, 224, 224)
        x = self.backbone(x).flatten(1)           # (B*9, 2048)
        x = x.view(b, seq, -1)                    # (B, 9, 2048)
        x, _ = self.lstm(x)                       # (B, 9, hidden_size)
        x = self.lstm_norm(x)                     # stable scale for downstream
        x = x[:, -1, :]                           # (B, hidden_size)
        x = self.fc(x)                            # (B, num_classes)
        return x


class Baseline5_stg2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(cfg.input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, 9, 12, input_size)
        batch, seq, players, feat = x.shape

        # Last timestep, pool over players
        last = x[:, -1, :, :]                             # (B, 12, input_size)
        pooled, _ = torch.max(last, dim=1)                # (B, input_size)

        return self.fc(pooled)

