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
        x = x[:, -1, :]                           # (B, hidden_size)
        x = self.fc(x)                            # (B, num_classes)
        return x


class Baseline5_stg2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # LSTM per player over time
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # Concat last ResNet feat + last LSTM hidden → classifier
        fc_in = cfg.input_size + cfg.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=2048) — pre-extracted features
        batch, seq, players, feat = x.shape

        # 1. LSTM per player over time
        x_p = x.permute(0, 2, 1, 3)                          # (B, 12, 9, 2048)
        x_p = x_p.contiguous().view(batch * players, seq, feat)  # (B*12, 9, 2048)
        lstm_out, _ = self.lstm(x_p)                          # (B*12, 9, hidden)

        # Last frame features
        last_resnet = x_p[:, -1, :]                           # (B*12, 2048)
        last_lstm = lstm_out[:, -1, :]                        # (B*12, hidden)
        combined = torch.cat([last_resnet, last_lstm], dim=1) # (B*12, 2048+hidden)

        # 2. Pool over players
        combined = combined.view(batch, players, -1)          # (B, 12, 2048+hidden)
        pooled, _ = torch.max(combined, dim=1)                # (B, 2048+hidden)

        # 3. Classify
        return self.fc(pooled)                                # (B, num_classes)
