import torch
import torch.nn as nn
from modeling.b5.model import Baseline5_stg1


class Baseline7(nn.Module):
    """
    Baseline 7: Full temporal model.
    Frozen B5 Stage 1 (ResNet + LSTM_1 per player) →
    Concat ResNet + LSTM features → Pool over players per frame →
    Trainable LSTM_2 over time → FC head.
    """

    def __init__(self, cfg):
        super().__init__()

        # --- Load and freeze B5 Stage 1 (backbone + lstm_1) ---
        stg1_cfg = type('Cfg', (), {
            'num_classes': cfg.num_classes_stg1,
            'hidden_size': cfg.hidden_size_stg1,
            'lstm_layers': cfg.lstm_layers_stg1,
            'dropout': cfg.dropout,
        })()
        stage1 = Baseline5_stg1(stg1_cfg)
        state_dict = torch.load(cfg.saved_stg1_path, map_location="cpu")
        stage1.load_state_dict(state_dict)

        self.backbone = stage1.backbone
        self.lstm_1 = stage1.lstm

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lstm_1.parameters():
            param.requires_grad = False

        # --- Pool over players per frame ---
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        # --- Layer norm for stability before LSTM_2 ---
        self.layer_norm = nn.LayerNorm(2048)

        # --- Trainable LSTM_2: learns group temporal dynamics ---
        self.lstm_2 = nn.LSTM(
            input_size=2048,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # --- FC head (matching reference architecture) ---
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, Seq=9, Players=12, C, H, W) from scenecrops_temporal
        b, seq, p, c, h, w = x.shape

        # 1. Frozen ResNet per player per frame
        x = x.view(b * seq * p, c, h, w)                    # (B*9*12, 3, 224, 224)
        with torch.no_grad():
            resnet_feat = self.backbone(x).flatten(1)        # (B*9*12, 2048)

        # 2. Frozen LSTM_1 per player over time
        lstm1_in = resnet_feat.view(b * p, seq, 2048)        # (B*12, 9, 2048)
        with torch.no_grad():
            lstm1_out, _ = self.lstm_1(lstm1_in)              # (B*12, 9, hidden_stg1)

        # 3. Concat ResNet + LSTM_1 features per player per frame
        resnet_temporal = resnet_feat.view(b * p, seq, 2048)  # (B*12, 9, 2048)
        combined = torch.cat([resnet_temporal, lstm1_out], dim=2)  # (B*12, 9, 2048+hidden_stg1)
        combined = combined.contiguous()

        # 4. Pool over players per frame
        combined = combined.view(b * seq, p, -1)              # (B*9, 12, 2048+hidden_stg1)
        pooled = self.pool(combined)                          # (B*9, 1, 2048)
        pooled = pooled.squeeze(dim=1)                        # (B*9, 2048)

        # 5. Trainable LSTM_2 over temporal sequence
        temporal_in = pooled.view(b, seq, 2048)               # (B, 9, 2048)
        temporal_in = self.layer_norm(temporal_in)             # (B, 9, 2048)
        lstm2_out, _ = self.lstm_2(temporal_in)                # (B, 9, hidden_size)
        last_hidden = lstm2_out[:, -1, :]                     # (B, hidden_size)

        # 6. Classify group activity
        out = self.fc(last_hidden)                            # (B, num_classes)
        return out
