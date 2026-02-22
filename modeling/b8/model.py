import torch
import torch.nn as nn


class Baseline8(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # LSTM for player-level temporal modeling
        self.player_lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True
        )

        # Adaptive pooling to summarize each team's features
        self.team_pool = nn.AdaptiveMaxPool1d(1)

        # LSTM for scene-level temporal modeling
        self.scene_lstm = nn.LSTM(
            input_size=cfg.hidden_size_player * 2,  # Combined team features
            hidden_size=cfg.hidden_size_frame,
            num_layers=1,
            batch_first=True
        )

        self.layer_norm_input = nn.LayerNorm(cfg.input_size)
        self.layer_norm_scene = nn.LayerNorm(cfg.hidden_size_player * 2)

        # Classifier for final prediction (deeper structure)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size_frame, cfg.hidden_fc1),
            nn.BatchNorm1d(cfg.hidden_fc1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(cfg.hidden_fc1, cfg.hidden_fc2),
            nn.BatchNorm1d(cfg.hidden_fc2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(cfg.hidden_fc2, cfg.hidden_fc3),
            nn.BatchNorm1d(cfg.hidden_fc3),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(cfg.hidden_fc3, cfg.num_classes)
        )

    def forward(self, x):
        # Input shape: (B, num_frames=9, num_players=12, feat=2048)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # 1. LSTM per player over time
        # Reshape: (B*12, 9, 2048)
        x = x.view(batch_size * num_players, num_frames, feature_dim)
        x = self.layer_norm_input(x)
        
        # Process each player with LSTM
        player_out, _ = self.player_lstm(x)  # (B*12, 9, hidden_player)
        player_out = player_out.contiguous()

        # 2. Reshape for team pooling at each frame
        # Shape: (B*9, 12, hidden_player)
        player_out = player_out.view(batch_size, num_players, num_frames, -1)
        player_out = player_out.permute(0, 2, 1, 3).contiguous()  # (B, 9, 12, hidden_player)
        player_out = player_out.view(batch_size * num_frames, num_players, -1)

        # Split into two teams (first 6 and last 6 players)
        team_1 = player_out[:, :6, :]  # (B*9, 6, hidden_player)
        team_2 = player_out[:, 6:, :]  # (B*9, 6, hidden_player)

        # Pool across players for each team
        team_1 = self.team_pool(team_1.permute(0, 2, 1)).squeeze(-1)  # (B*9, hidden_player)
        team_2 = self.team_pool(team_2.permute(0, 2, 1)).squeeze(-1)  # (B*9, hidden_player)

        # Concatenate team features
        scene_input = torch.cat([team_1, team_2], dim=1)  # (B*9, hidden_player * 2)

        # 3. Scene LSTM over temporal sequence
        scene_input = scene_input.view(batch_size, num_frames, -1)  # (B, 9, hidden_player*2)
        scene_input = self.layer_norm_scene(scene_input)
        
        scene_out, _ = self.scene_lstm(scene_input)  # (B, 9, hidden_frame)

        # 4. Classify (last frame representation)
        final = scene_out[:, -1, :]  # (B, hidden_frame)
        return self.classifier(final)
