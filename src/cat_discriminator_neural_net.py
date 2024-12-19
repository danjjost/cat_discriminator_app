import torch

import torch.nn as nn
import torch.nn.functional as F

class CatDiscriminatorNeuralNet(nn.Module):
    def __init__(self, debug = False) -> None:
        super().__init__()
        self.debug = debug
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=10) # 10x10 kernels, moving across the 3 input channels (rgb), outputting 12 channels. The new (12) channels will be 504x504
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 12 feature maps, 252x252
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5) # 12 input feature maps, 24 output feature maps. The new (24) channels will be 248x248
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # this declaration is unnecessary, since we could reuse the first pool. The output will be 24 feature maps 124x124
        
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5) # 24 input feature maps, 48 output feature maps. The new (48) channels will be 120x120
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # The output will be 48 feature maps 70x70
        
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3) # 48 input feature maps, 96 output feature maps. The new (96) channels will be 68x68
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # The output will be 96 feature maps 34x34

        # These are the fully connected layers
        self.fc1 = nn.Linear(96 * 29 * 29, 1624)
        self.fc2 = nn.Linear(1624, 864)
        self.fc3 = nn.Linear(864, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x) 
        x = F.relu(x) 
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flattening from feature maps to fully connected layers

        x = torch.flatten(x, 1) 
        
        # Fully connected layers

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) 
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)

        return x