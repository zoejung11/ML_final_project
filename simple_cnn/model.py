import torch
import torch.nn as nn

class SimpleSegmentationCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Feature Extraction Layer 1:
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        logits = self.final_conv(x3)
        output = self.sigmoid(logits)
        return output