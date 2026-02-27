import torch.nn as nn


class Baseline7(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Player-level LSTM (learns temporal features from raw CNN)
        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True,
        )

        # Scene-level LSTM (operates on pooled player representations)
        self.lstm2 = nn.LSTM(
            input_size=cfg.hidden_size_player,
            hidden_size=cfg.hidden_size_scene,
            num_layers=1,
            batch_first=True,
        )

        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size_scene, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=2048)
        batch, seq, players, feat = x.shape

        # LSTM1: per-player temporal modeling
        x = x.view(batch * players, seq, feat)               # (B*12, 9, 2048)
        x, _ = self.lstm1(x)                                  # (B*12, 9, hidden_player)

        x = x.view(batch, players, seq, -1)                   # (B, 12, 9, hidden_player)

        # Max pool over players per frame
        x = x.permute(0, 2, 3, 1).contiguous()               # (B, 9, hidden_player, 12)
        x = x.view(batch * seq, -1, players)                  # (B*9, hidden_player, 12)
        x = self.adaptive_max_pool(x).squeeze(-1)             # (B*9, hidden_player)
        x = x.view(batch, seq, -1)                            # (B, 9, hidden_player)

        # LSTM2: scene-level temporal modeling
        x, _ = self.lstm2(x)                                  # (B, 9, hidden_scene)
        x = x[:, -1, :]                                      # (B, hidden_scene)

        return self.fc(x)
