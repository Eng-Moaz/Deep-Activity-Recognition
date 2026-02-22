import torch.nn as nn


class Baseline4(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # LSTM over frame sequence
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, feat=2048) — pre-extracted frame features
        lstm_out, _ = self.lstm(x)            # (B, 9, hidden_size)
        last_hidden = lstm_out[:, -1, :]      # (B, hidden_size)
        return self.fc(last_hidden)           # (B, num_classes)