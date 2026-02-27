import torch
import torch.nn as nn


class Baseline8(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Player-level LSTM (learns temporal features from raw CNN)
        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True,
        )

        # Scene-level LSTM (operates on concatenated team representations)
        self.lstm2 = nn.LSTM(
            input_size=cfg.hidden_size_player * 2,
            hidden_size=cfg.hidden_size_scene,
            num_layers=1,
            batch_first=True,
        )

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

        # Split into two teams
        team1 = x[:, :6, :, :]                                # (B, 6, 9, hidden_player)
        team2 = x[:, 6:, :, :]                                # (B, 6, 9, hidden_player)

        # Max pool over players within each team
        team1 = torch.max(team1, dim=1)[0]                    # (B, 9, hidden_player)
        team2 = torch.max(team2, dim=1)[0]                    # (B, 9, hidden_player)

        # Concatenate team representations
        scene_input = torch.cat([team1, team2], dim=2)        # (B, 9, hidden_player*2)

        # LSTM2: scene-level temporal modeling
        scene_out, _ = self.lstm2(scene_input)                # (B, 9, hidden_scene)
        scene_out = scene_out[:, -1, :]                       # (B, hidden_scene)

        return self.fc(scene_out)
