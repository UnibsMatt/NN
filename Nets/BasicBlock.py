import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        in_channel: canali in input
        out_channel: canali in uscita
        """
        super(BasicBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=3,
                                padding=0)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel)
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu_activation(x)
        return x


class PreProcess(nn.Module):
    def __init__(self):
        """
        Blocco di pre-processing delle immagini. Prende il tensore in ingresso nella forma
        (batch, width, height, channel), lo permuta e lo normalizza tra 0 e 1.
        """
        super(PreProcess, self).__init__()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float()
        x = x.div(255.)
        return x
