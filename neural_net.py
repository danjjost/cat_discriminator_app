import torch

import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 12, 5) # 5x5 kernel, moving across the 3 input channels (rgb), outputting 12 channels. The new (12) channels will be 28x28
        self.pool1 = nn.MaxPool2d(2, 2) # (12, 14, 14)
        
        self.conv2 = nn.Conv2d(12, 24, 5) # 12 input feature maps, 24 output feature maps. The new (24) channels will be 10x10
        self.pool2 = nn.MaxPool2d(2, 2) # this is unnecessary since we could reuse the first pool. The output will be 24x5x5
        
        # These are the fully connected layers
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # 3x32x32 -> 12x28x28
        x = F.relu(x) # 12x28x28 Bounded positive
        x = self.pool1(x) # 12x28x28 -> 12x14x14
        
        x = self.conv2(x) # 12x14x14 -> 24x10x10
        x = F.relu(x) # 24x10x10
        x = self.pool2(x) # 24x10x10 -> 24x5x5
        
        # Flatten the feature maps into a 1D tensor
        x = torch.flatten(x, 1)
        
        # take the 1D tensor and pass it through the fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x