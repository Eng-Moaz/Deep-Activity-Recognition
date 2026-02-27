import torch
import torch.nn as nn


class Baseline7(nn.Module):
    """B7: Player LSTM + max pool (with CNN+LSTM concat) + Scene LSTM."""

    def __init__(self, cfg):
        super().__init__()

        # Player-level LSTM
        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size_player,
            num_layers=1,
            batch_first=True,
        )

        # Scene-level LSTM (input = CNN features + LSTM1 output, after pooling)
        self.lstm2 = nn.LSTM(
            input_size=cfg.input_size + cfg.hidden_size_player,
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
        x_flat = x.view(batch * players, seq, feat)           # (B*12, 9, 2048)
        lstm1_out, _ = self.lstm1(x_flat)                      # (B*12, 9, hidden_player)

        # Reshape back
        lstm1_out = lstm1_out.view(batch, players, seq, -1)    # (B, 12, 9, hidden_player)
        x_orig = x.permute(0, 2, 1, 3)                        # (B, 12, 9, 2048)

        # Concatenate CNN features with LSTM1 output (paper Eq. 7: P_tk = x_tk ⊕ h_tk)
        combined = torch.cat([x_orig, lstm1_out], dim=3)       # (B, 12, 9, 2048 + hidden_player)
        combined = combined.permute(0, 2, 3, 1).contiguous()   # (B, 9, feat_combined, 12)

        # Max pool over players per frame
        B_seq = batch * seq
        feat_combined = feat + self.lstm1.hidden_size
        combined = combined.view(B_seq, feat_combined, players) # (B*9, feat_combined, 12)
        pooled = self.adaptive_max_pool(combined).squeeze(-1)   # (B*9, feat_combined)
        pooled = pooled.view(batch, seq, feat_combined)         # (B, 9, feat_combined)

        # LSTM2: scene-level temporal modeling
        scene_out, _ = self.lstm2(pooled)                      # (B, 9, hidden_scene)
        scene_out = scene_out[:, -1, :]                        # (B, hidden_scene)

        return self.fc(scene_out)
