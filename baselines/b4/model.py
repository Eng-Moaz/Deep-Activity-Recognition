from baselines.b1.model import Baseline1
import torch
import torch.nn as nn

class Baseline4(nn.Module):
    def __init__(self,trained_resnet_path,hidden_size,lstm_layers,num_classes):
        super().__init__()
        #Load the pre-trained model
        self.baseline1 = Baseline1(num_classes,0.5)
        state_dict = torch.load(trained_resnet_path, map_location="cpu")
        self.baseline1.load_state_dict(state_dict)

        #Remove the last layer
        full_resnet = self.baseline1.backbone
        modules = list(full_resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        #Freeze the network
        for param in self.backbone.parameters():
            param.requires_grad = False

        #LSTM model
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        #Last fc Layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        batch_size , sequence_length , ch , h, w = x.size()
        spatial_input = x.view(batch_size*sequence_length,ch,h,w)
        features = self.backbone(spatial_input)
        features = features.view(features.size(0), -1)
        temporal_in = features.view(batch_size,sequence_length,-1)
        temporal_out , _ = self.lstm(temporal_in)
        final_temporal = temporal_out[:, -1, :]
        final_out = self.fc(final_temporal)
        return final_out