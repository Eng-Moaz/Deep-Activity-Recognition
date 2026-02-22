import torch
import torch.nn as nn


class Baseline7(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # LSTM per player over time
        self.player_lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, num_frames=9, num_players=12, feat=2048)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # Reshape: each player gets its own temporal sequence
        x = x.view(batch_size * num_players, num_frames, feature_dim)

        # LSTM per player — take last hidden state
        _, (h_n, _) = self.player_lstm(x)
        x = h_n[-1]  # (B*12, hidden)

        # Reshape back to (B, 12, hidden) then pool over players
        x = x.view(batch_size, num_players, -1)
        pooled = torch.max(x, dim=1)[0]  # (B, hidden)

        # Classify
        return self.classifier(pooled)
