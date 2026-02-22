import torch
import torch.nn as nn


class Baseline6(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Max pool over players
        self.pool = nn.AdaptiveMaxPool1d(1)

        # LSTM over frame sequence (after pooling players)
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # FC classifier
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=2048) — pre-extracted features
        batch, seq, players, feat = x.shape

        # 1. Pool over players per frame
        x = x.view(batch * seq, players, feat)       # (B*9, 12, 2048)
        x = x.permute(0, 2, 1)                       # (B*9, 2048, 12)
        x = self.pool(x).squeeze(-1)                  # (B*9, 2048)
        x = x.view(batch, seq, feat)                  # (B, 9, 2048)

        # 2. LSTM over temporal sequence
        lstm_out, _ = self.lstm(x)                    # (B, 9, hidden)
        last_hidden = lstm_out[:, -1, :]              # (B, hidden)

        # 3. Classify
        return self.fc(last_hidden)                   # (B, num_classes)