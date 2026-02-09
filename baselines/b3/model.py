import torch
import torch.nn as nn
import torchvision.models as models

class Baseline3_stg1(nn.Module):
    def __init__(self,num_classes,dropout):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features,num_classes)
        )
    def forward(self,x):
        return self.backbone(x)

class Baseline3_stg2(nn.Module):
    def __init__(self,num_classes,dropout,saved_resnet_path):
        super().__init__()
        # Load the pre-trained model
        self.baseline3_stg1 = Baseline3_stg1(9,0.5)
        state_dict = torch.load(saved_resnet_path, map_location="cpu")
        self.baseline3_stg1.load_state_dict(state_dict)

        # Remove the last layer
        full_resnet = self.baseline3_stg1.backbone
        modules = list(full_resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # Freeze the network
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.scene_fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024,num_classes)
        )

    def forward(self, x):
        # Merge batch_size and num_players
        batch_size, num_players, ch, h, w = x.shape
        spatial_input = x.view(batch_size * num_players, ch, h, w)  # (B, 12, 3, 224, 224) -> (B*12, 3, 224, 224)

        # Backbone forward
        features = self.backbone(spatial_input)  # [B*12, 2048, 1, 1]
        features = features.flatten(1)  # (B*12 x 2048)

        # Reshape back to separate players
        features = features.view(batch_size, num_players, -1)  # (B*12 x 2048) -> (B x 12 x 2048)

        # MaxPool across players
        scene_features, _ = torch.max(features, dim=1)  # Take max across the 12 players -> (B x 2048)

        # Classify the Scene
        final_out = self.scene_fc(scene_features)

        return final_out