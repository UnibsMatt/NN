from Nets.Networks import *
import torch
import numpy as np

model = Net()

inputs = torch.randint(0, 255, (64, 1, 28, 28)).float()

for i in range(1000):
    ou = model(inputs)
    print(ou)
