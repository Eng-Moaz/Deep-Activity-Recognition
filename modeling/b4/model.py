from modeling.b1.model import Baseline1
import torch
import torch.nn as nn

class Baseline4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Load the pre-trained B1 model
        b1_cfg = type('Cfg', (), {'num_classes': cfg.num_classes, 'dropout': cfg.dropout})()
        self.baseline1 = Baseline1(b1_cfg)
        state_dict = torch.load(cfg.trained_resnet_path, map_location="cpu")
        self.baseline1.load_state_dict(state_dict)

        # Remove the last layer
        full_resnet = self.baseline1.backbone
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
            batch_first=True
        )

        # Last fc Layer
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, x):
        # Merge batch_size and seq_length
        batch_size, sequence_length, ch, h, w = x.shape
        spatial_input = x.view(batch_size * sequence_length, ch, h, w)  # (B, 9, 3, 224, 224) -> (B*9, 3, 224, 224)

        # Backbone forward
        features = self.backbone(spatial_input)
        features = features.flatten(1)  # (B*9 x 2048 x 1 x 1) -> (72 x 2048)

        # Reshape for LSTM
        temporal_in = features.view(batch_size, sequence_length, -1)  # (B*9 x 2048) -> (B x 9 x 2048)

        # LSTM forward
        temporal_out, _ = self.lstm(temporal_in)
        final_temporal = temporal_out[:, -1, :]  # Take the last time step

        final_out = self.fc(final_temporal)

        return final_out