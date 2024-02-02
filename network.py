import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torchvision import datasets


class Net(nn.Module):
    
    step = 0
    
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 
	    # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5) # input = 3 por que nossa imagem eh RGB, output  = 6 (6 feature maps),  kernel = 5 (ou 5x5)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 
        self.conv2 = nn.Conv2d(6, 16, 5) # o input deste layer eh o output do layer anterior, ou seja 6.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# output do layer anterior x 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x