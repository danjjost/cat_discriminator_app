import torch

import torch.nn as nn
import torch.nn.functional as F

class CatDiscriminatorNeuralNet(nn.Module):
    def __init__(self, debug = False) -> None:
        super().__init__()
        self.debug = debug
        
        self.conv1 = nn.Conv2d(3, 12, 5) # 5x5 kernel, moving across the 3 input channels (rgb), outputting 12 channels. The new (12) channels will be 124x124
        self.pool1 = nn.MaxPool2d(2, 2) # 12 feature maps, 62x62
        
        self.conv2 = nn.Conv2d(12, 24, 5) # 12 input feature maps, 24 output feature maps. The new (24) channels will be 58x58
        self.pool2 = nn.MaxPool2d(2, 2) # this is unnecessary since we could reuse the first pool. The output will be 24 feature maps 29x29
        
        self.conv3 = nn.Conv2d(24, 48, 5) # 24 input feature maps, 48 output feature maps. The new (48) channels will be 25x25
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True) # The output will be 48 feature maps 13x13
        
        self.conv4 = nn.Conv2d(48, 96, 5) # 48 input feature maps, 96 output feature maps. The new (96) channels will be 9x9
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True) # The output will be 96 feature maps 5x5

        # These are the fully connected layers
        self.fc1 = nn.Linear(96 * 5 * 5, 1200)
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, 120)
        self.fc4 = nn.Linear(120, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # 3x128x128 -> 12x124x124
        x = F.relu(x) # 12x124x124 Bounded positive
        x = self.pool1(x) # 12x124x124 -> 12x62x62
        
        if self.debug:
            print('After pool1:', x.shape)

        x = self.conv2(x) # 12x62x62 -> 24x58x58
        x = F.relu(x) # 24x58x58
        x = self.pool2(x) # 24x58x58 -> 24x29x29

        if self.debug:
            print('After pool2:', x.shape)
        
        x = self.conv3(x) # 24x29x29 -> 48x25x25
        x = F.relu(x) # 48x25x25
        x = self.pool3(x) # 48x25x25 -> 48x13x13
        if self.debug:
            print('After pool3:', x.shape)
        
        x = self.conv4(x) # 48x13x13 -> 96x9x9
        x = F.relu(x)
        x = self.pool4(x) # 96x9x9 -> 96x5x5
        if self.debug:
            print('After pool4:', x.shape)
        
        # Flatten the feature maps into a 1D tensor
        x = torch.flatten(x, 1) # 96x5x5 -> 96*5*5 (2400)
        if self.debug:
            print(x.shape)
        
        # take the 1D tensor and pass it through the fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        return x