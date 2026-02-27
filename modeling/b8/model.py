import torch
import torch.nn as nn


class Baseline8(nn.Module):
    """B8: Player LSTM + team pooling (with CNN+LSTM concat) + 2-layer Scene LSTM."""

    def __init__(self, cfg):
        super().__init__()

        # Player-level LSTM
        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True,
        )

        # Scene-level LSTM (input = 2 teams × (CNN + LSTM1) concatenated)
        scene_input_size = (cfg.input_size + cfg.hidden_size_player) * 2
        self.lstm2 = nn.LSTM(
            input_size=scene_input_size,
            hidden_size=cfg.hidden_size_scene,
            num_layers=cfg.lstm2_layers,
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
        x_flat = x.view(batch * players, seq, feat)           # (B*12, 9, 2048)
        lstm1_out, _ = self.lstm1(x_flat)                      # (B*12, 9, hidden_player)

        # Reshape back
        lstm1_out = lstm1_out.view(batch, players, seq, -1)    # (B, 12, 9, hidden_player)
        x_orig = x.permute(0, 2, 1, 3)                        # (B, 12, 9, 2048)

        # Concatenate CNN features with LSTM1 output (paper Eq. 7)
        combined = torch.cat([x_orig, lstm1_out], dim=3)       # (B, 12, 9, 2048 + hidden_player)

        # Split into two teams
        team1 = combined[:, :6, :, :]                          # (B, 6, 9, feat_combined)
        team2 = combined[:, 6:, :, :]                          # (B, 6, 9, feat_combined)

        # Max pool over players within each team
        team1 = torch.max(team1, dim=1)[0]                     # (B, 9, feat_combined)
        team2 = torch.max(team2, dim=1)[0]                     # (B, 9, feat_combined)

        # Concatenate team representations
        scene_input = torch.cat([team1, team2], dim=2)         # (B, 9, feat_combined * 2)

        # LSTM2: scene-level temporal modeling
        scene_out, _ = self.lstm2(scene_input)                 # (B, 9, hidden_scene)
        scene_out = scene_out[:, -1, :]                        # (B, hidden_scene)

        return self.fc(scene_out)
