import torch
import torch.nn as nn


class Baseline8(nn.Module):
    """LSTM per player → split teams → pool each team → concat → scene LSTM → FC."""

    def __init__(self, cfg):
        super().__init__()

        # LSTM for player-level temporal modeling
        self.player_lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True,
        )

        # Pooling per team
        self.team_pool = nn.AdaptiveMaxPool1d(1)

        # Scene LSTM over concatenated team features
        self.scene_lstm = nn.LSTM(
            input_size=cfg.hidden_size_player * 2,  # two teams concatenated
            hidden_size=cfg.hidden_size_frame,
            num_layers=1,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size_frame, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, num_frames=9, num_players=12, feat=2048)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # 1. LSTM per player over time
        x = x.view(batch_size * num_players, num_frames, feature_dim)
        player_out, _ = self.player_lstm(x)  # (B*12, 9, hidden_player)

        # Reshape to (B*9, 12, hidden_player) for per-frame team pooling
        player_out = player_out.view(batch_size, num_players, num_frames, -1)
        player_out = player_out.permute(0, 2, 1, 3).contiguous()  # (B, 9, 12, hidden_player)
        player_out = player_out.view(batch_size * num_frames, num_players, -1)

        # 2. Split into two teams and pool each
        team_1 = player_out[:, :6, :]  # (B*9, 6, hidden_player)
        team_2 = player_out[:, 6:, :]  # (B*9, 6, hidden_player)

        team_1 = self.team_pool(team_1.permute(0, 2, 1)).squeeze(-1)  # (B*9, hidden_player)
        team_2 = self.team_pool(team_2.permute(0, 2, 1)).squeeze(-1)  # (B*9, hidden_player)

        # 3. Concatenate team representations
        scene_input = torch.cat([team_1, team_2], dim=1)  # (B*9, hidden_player*2)
        scene_input = scene_input.view(batch_size, num_frames, -1)  # (B, 9, hidden_player*2)

        # 4. Scene LSTM over temporal sequence
        scene_out, _ = self.scene_lstm(scene_input)  # (B, 9, hidden_frame)
        final = scene_out[:, -1, :]  # (B, hidden_frame)

        # 5. Classify
        return self.classifier(final)
