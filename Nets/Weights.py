import torch
import torch.nn as nn


def uniform_weights_initialization(model):
    for module in model.modules():
        if type(module) == nn.Conv2d:
            torch.nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)
