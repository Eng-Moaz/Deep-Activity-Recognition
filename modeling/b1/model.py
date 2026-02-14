import torch.nn as nn
import torchvision.models as models

class Baseline1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(self.backbone.fc.in_features, cfg.num_classes)
        )
    def forward(self,x):
        return self.backbone(x)
