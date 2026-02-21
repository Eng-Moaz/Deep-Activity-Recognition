import torch
import torch.nn as nn
import torchvision.models as models


class Baseline5_stg1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, Seq=9, C, H, W) — one player's crops over time
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)             # (B*9, 3, 224, 224)
        x = self.backbone(x).flatten(1)           # (B*9, 2048)
        x = x.view(b, seq, -1)                    # (B, 9, 2048)
        x, _ = self.lstm(x)                       # (B, 9, hidden_size)
        x = x[:, -1, :]                           # (B, hidden_size)
        x = self.fc(x)                            # (B, num_classes)
        return x


class Baseline5_stg2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load full Stage 1 and freeze backbone + LSTM
        stg1_hidden = getattr(cfg, 'hidden_size_stg1', cfg.hidden_size)
        stg1_cfg = type('Cfg', (), {
            'num_classes': cfg.num_classes_stg1,
            'hidden_size': stg1_hidden,
            'lstm_layers': cfg.lstm_layers,
            'dropout': cfg.dropout,
        })()
        stage1 = Baseline5_stg1(stg1_cfg)
        state_dict = torch.load(cfg.saved_stg1_path, map_location="cpu")
        stage1.load_state_dict(state_dict)

        self.backbone = stage1.backbone
        self.lstm = stage1.lstm

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Group classifier (operates on pooled LSTM features)
        fc_in = stg1_hidden + 2048
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(512, cfg.num_classes),
        )

    def forward(self, x):
        # x: (B, Seq=9, Players=12, C, H, W) from scenecrops_temporal
        b, seq, p, c, h, w = x.shape

        # 1. CNN per player per frame (frozen)
        x = x.view(b * seq * p, c, h, w)              # (B*9*12, 3, 224, 224)
        with torch.no_grad():
            resnet_features = self.backbone(x).flatten(1)            # (B*9*12, 2048)

        # 2. LSTM per player over time (frozen)
        lstm_in = resnet_features.view(b * p, seq, 2048)                  # (B*12, 9, 2048)
        with torch.no_grad():
            lstm_out, _ = self.lstm(lstm_in)                    # (B*12, 9, hidden_size)

        last_frame_resnet = lstm_in[:, -1, :]  # (Batch*Players, 2048)
        last_frame_lstm = lstm_out[:, -1, :] # (B*12, hidden_size)

        combined_features = torch.cat([last_frame_resnet, last_frame_lstm], dim=1)

        # 3. Pool over players
        combined_features = combined_features.view(b, p, -1)  # (Batch, 12, 2048+Hidden)
        pooled_features, _ = torch.max(combined_features, dim=1) # (Batch, 2048+Hidden)

        # 4. Classify group activity
        x = self.fc(pooled_features) # (B, num_classes)
        return x
