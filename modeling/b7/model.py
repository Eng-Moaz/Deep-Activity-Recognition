import torch
import torch.nn as nn


class Baseline7(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # LSTM per player over time
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # Max pool over players
        self.pool = nn.AdaptiveMaxPool1d(1)

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

        # 1. LSTM per player over time
        x = x.permute(0, 2, 1, 3)                       # (B, 12, 9, 2048)
        x = x.contiguous().view(batch * players, seq, feat)  # (B*12, 9, 2048)
        _, (h_n, _) = self.lstm(x)                       # h_n: (layers, B*12, hidden)
        x = h_n[-1]                                      # (B*12, hidden)

        # 2. Pool over players
        x = x.view(batch, players, -1)                   # (B, 12, hidden)
        x = x.permute(0, 2, 1)                           # (B, hidden, 12)
        x = self.pool(x).squeeze(-1)                     # (B, hidden)

        # 3. Classify
        return self.fc(x)                                # (B, num_classes)
