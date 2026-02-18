from modeling.b3.model import Baseline3_stg1
import torch.nn as nn
import torch

class Baseline6(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load the pre-trained Stage 1 model (needs stg1's num_classes to load weights)
        stg1_cfg = type('Cfg', (), {'num_classes': cfg.num_classes_stg1, 'dropout': cfg.dropout})()
        self.baseline3_stg1 = Baseline3_stg1(stg1_cfg)
        state_dict = torch.load(cfg.saved_resnet50_path, map_location="cpu")
        self.baseline3_stg1.load_state_dict(state_dict)

        # Remove the last layer
        full_resnet = self.baseline3_stg1.backbone
        modules = list(full_resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # Freeze the network
        for param in self.backbone.parameters():
            param.requires_grad = False

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(256, cfg.num_classes),
        )

    def forward(self, x):
        # Input -> (Batch, frames(9), players(12), C, H, W)
        batch, seq, players, ch, h, w = x.shape

        # Extract CNN features (frozen)
        x = x.view(batch * seq * players, ch, h, w)        # (B*9*12, 3, 224, 224)
        with torch.no_grad():
            feature_vectors = self.backbone(x)              # (B*9*12, 2048, 1, 1)
        feature_vectors = feature_vectors.flatten(1)        # (B*9*12, 2048)
        feature_vectors = feature_vectors.view(batch, seq, players, 2048)

        # Max-pool over players
        temporal_input, _ = torch.max(feature_vectors, dim=2)   # (B, 9, 2048)

        # LSTM
        lstm_out, _ = self.lstm(temporal_input)             # (B, 9, hidden_size)
        last_hidden = lstm_out[:, -1, :]                    # (B, hidden_size)

        # Classify
        out = self.fc(last_hidden)                          # (B, num_classes)
        return out