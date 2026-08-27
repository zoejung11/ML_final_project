import torch
import torch.nn as nn

class DeeperSegmentationCNN(nn.Module):
    def __init__(self, input_channels:int = 1, output_channels:int = 1):
        super().__init__()
        
        self.network = nn.Sequential(
          nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
          nn.ReLU(),
            
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(),
           
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
          nn.ReLU(),
          
          nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(),
          
          nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
          nn.ReLU(), 
          
          nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=3, padding=1),
          nn.Sigmoid()         
        )
    
    def forward(self, x):
        return self.network(x)