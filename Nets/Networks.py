import torch.nn as nn
from Nets.BasicBlock import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pre_process = PreProcess()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            BasicBlock(1, 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            BasicBlock(4, 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 5 * 5, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.pre_process(x)

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x