## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # (32 , 220 , 220)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool2d(2,2)  # (16 , 110 , 110)
        self.conv2 = nn.Conv2d(32, 64 , 5) # (16 , 106 , 106)
        self.bn2 = nn.BatchNorm1d(64)
        # Pool (8 , 53  ,53)
        self.conv3 = nn.Conv2d(64 , 128 , 5) # (8 , 49 , 49)
        self.bn3 = nn.BatchNorm1d(128)
        # Pool (4 , 24  ,24)
        self.conv4 = nn.Conv2d(128 , 256 , 5) # (4 , 20 , 20)
        self.bn4 = nn.BatchNorm1d(256)
        # Pool (2 , 10 , 10)
        self.fc1 = nn.Linear(2 * 10 * 10 , 256)
        
        self.fc_drp = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256 , 136)
        
    def forward(self , x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0) , -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc_drp(x)
        x = self.fc2(x)
        
        return x
                      
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
       