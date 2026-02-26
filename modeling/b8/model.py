import torch
import torch.nn as nn


class Baseline8(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Team pooling
        self.team_pool = nn.AdaptiveMaxPool1d(1)

        # Scene-level LSTM (input = two teams concatenated)
        self.scene_lstm = nn.LSTM(
            input_size=cfg.input_size * 2,
            hidden_size=cfg.hidden_size_frame,
            num_layers=1,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(cfg.input_size * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear((cfg.input_size * 2) + cfg.hidden_size_frame, cfg.hidden_fc),
            nn.BatchNorm1d(cfg.hidden_fc),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_fc, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, seq=9, players=12, feat=input_size)
        batch, seq, players, feat = x.shape

        # Split into two teams and pool each
        x = x.view(batch * seq, players, feat)

        team_1 = x[:, :6, :]                                    # (B*9, 6, feat)
        team_2 = x[:, 6:, :]                                    # (B*9, 6, feat)

        team_1 = self.team_pool(team_1.permute(0, 2, 1)).squeeze(-1)  # (B*9, feat)
        team_2 = self.team_pool(team_2.permute(0, 2, 1)).squeeze(-1)  # (B*9, feat)

        # Concatenate team representations
        scene_input = torch.cat([team_1, team_2], dim=1)         # (B*9, feat*2)
        scene_input = scene_input.view(batch, seq, -1)           # (B, 9, feat*2)
        scene_input = self.layer_norm(scene_input)

        # Scene LSTM over temporal sequence
        scene_out, _ = self.scene_lstm(scene_input)              # (B, 9, hidden_frame)
        
        # Concatenate scene representations from CNN/LSTM1 and LSTM2 over time, take the last frame
        x_concat = torch.cat([scene_input, scene_out], dim=2)    # (B, 9, input_size*2 + hidden_frame)
        final = x_concat[:, -1, :]                               # (B, input_size*2 + hidden_frame)

        return self.classifier(final)
