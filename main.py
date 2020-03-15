from Nets.Networks import *
from Nets.BasicBlock import *
from Nets.Weights import *
import torch
import numpy as np

model = Net()
torch.manual_seed(123)
inputs = torch.randint(0, 255, (64, 28, 28, 1)).float()

ao = model(inputs)
uniform_weights_initialization(model)

ao1 = model(inputs)

for i in range(1000):
    ou = model(inputs)
    print(ou)
