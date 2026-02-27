import torch
import torch.nn as nn


class Baseline7(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Pool over players
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Scene-level LSTM over the temporal sequence
        self.scene_lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.lstm_dropout = nn.Dropout(cfg.dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.input_size + cfg.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=input_size)
        batch, seq, players, feat = x.shape

        # Pool over players per frame
        x = x.view(batch * seq, players, feat)      # (B*9, 12, feat)
        x = x.permute(0, 2, 1)                      # (B*9, feat, 12)
        x = self.pool(x).squeeze(-1)                 # (B*9, feat)
        x = x.view(batch, seq, feat)                 # (B, 9, feat)

        # Scene LSTM over temporal sequence
        lstm_out, _ = self.scene_lstm(x)             # (B, 9, hidden)
        lstm_out = self.lstm_dropout(lstm_out)

        # Temporal mean pooling over all timesteps
        x_mean = x.mean(dim=1)                       # (B, feat)
        lstm_mean = lstm_out.mean(dim=1)              # (B, hidden)
        combined = torch.cat([x_mean, lstm_mean], dim=1)  # (B, feat + hidden)

        return self.classifier(combined)
