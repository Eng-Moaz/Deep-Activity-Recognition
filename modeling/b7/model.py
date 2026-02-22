import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerAttention(nn.Module):
    """Learnable attention pooling over players."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x):
        # x: (B, players, hidden)
        scores = self.attn(x)                           # (B, players, 1)
        weights = F.softmax(scores, dim=1)              # (B, players, 1)
        return (x * weights).sum(dim=1)                 # (B, hidden)


class Baseline7(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.feat_dropout = nn.Dropout(cfg.feat_dropout)

        # Bidirectional LSTM per player over time
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        # Bidirectional doubles the output size
        lstm_out_size = cfg.hidden_size * 2

        # Attention pooling over players
        self.player_attn = PlayerAttention(lstm_out_size)

        # FC classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=2048) — pre-extracted features
        batch, seq, players, feat = x.shape

        # 0. Feature-level dropout (regularization)
        x = self.feat_dropout(x)

        # 1. Bidirectional LSTM per player over time
        x = x.permute(0, 2, 1, 3)                           # (B, 12, 9, 2048)
        x = x.contiguous().view(batch * players, seq, feat)  # (B*12, 9, 2048)
        lstm_out, _ = self.lstm(x)                           # (B*12, 9, hidden*2)
        x = lstm_out.mean(dim=1)                             # (B*12, hidden*2)  — mean over time

        # 2. Attention pool over players
        x = x.view(batch, players, -1)                       # (B, 12, hidden*2)
        x = self.player_attn(x)                              # (B, hidden*2)

        # 3. Classify
        return self.fc(x)                                    # (B, num_classes)
