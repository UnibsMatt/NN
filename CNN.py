import torch
import torch.nn as nn
import numpy as np


class ProtoNet(nn.Module):
    def __init__(self, x_dim=1, h_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            basic_conv2d_block(x_dim, h_dim),
            basic_conv2d_block(h_dim, h_dim),
            basic_conv2d_block(h_dim, h_dim),
            basic_conv2d_block(h_dim, z_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class CNNOcr(nn.Module):
    def __init__(self, input_shape=(50, 50, 3), class_number=10):
        super(CNNOcr, self).__init__()
        self.classes = class_number
        self.input_shape = input_shape
        self.filter = 16

        self.pre_process = PreProcess()

        self.first_avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2d_1 = basic_conv2d_block(input_shape[2], self.filter)
        self.conv2d_2 = basic_conv2d_block(self.filter, self.filter*3)
        self.conv2d_3 = basic_conv2d_block(self.filter * 3, self.filter * 4)
        self.conv2d_4 = basic_conv2d_block(self.filter * 4, self.filter * 5)

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(.15)
        self.fc = nn.Linear(37632, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.pre_process(x)
        x = self.first_avg_pooling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        confidence, pred_class = torch.max(x, dim=1)
        confid = confidence.unsqueeze(dim=1)
        pred = pred_class.unsqueeze(dim=1)
        return torch.cat((confid.float(), pred.float(), x.float()), dim=1)


class CNNPreProcessed(torch.nn.Module):
    def __init__(self, input_shape=(470, 470, 3), class_number=2):
        super(CNNPreProcessed, self).__init__()
        self.classes = class_number
        self.input_shape = input_shape
        self.filter = 24
        self.pre_process = PreProcess()
        self.first_avg_pooling = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2d_1 = basic_conv2d_block(input_shape[2], self.filter)
        self.conv2d_2 = basic_conv2d_block(self.filter, self.filter*2)
        self.conv2d_3 = basic_conv2d_block(self.filter * 2, self.filter * 4)
        self.conv2d_4 = basic_conv2d_block(self.filter * 4, self.filter * 8)
        self.conv2d_5 = basic_conv2d_block(self.filter * 8, self.filter * 10)
        self.conv2d_6 = basic_conv2d_block(self.filter * 10, self.filter * 14)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout = nn.Dropout2d(.2)
        self.fc = nn.Linear(3072, self.classes)
        self.softmax = nn.Softmax(dim=1)
        self.grad = None

    def heat_hook(self, grad):
        self.grad = grad

    def forward(self, x, y):

        x = self.pre_process(x)

        x = self.first_avg_pooling(x)

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)

        #x = self.conv2d_5(x)
        #x = self.conv2d_6(x)
        x.register_hook(self.heat_hook)

        x = self.max_pooling(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        x.backward(x, retain_graph=True)

        confidence, pred_class = torch.max(x, dim=1)

        heat = nn.Conv2d(192, 1, kernel_size=1)(self.grad.cpu())
        heat = heat.reshape((self.grad.shape[0], 9, 9))

        return confidence, pred_class, x[:,0], x[:,1], heat


class Short_CNN(torch.nn.Module):
    def __init__(self, input_shape=(3, 128, 128), class_number=2):
        super(Short_CNN, self).__init__()
        self.classes = class_number
        self.input_shape = input_shape
        self.filter = 16
        self.first_avg_pooling = nn.AvgPool2d(kernel_size=4, stride=2)
        self.conv2d_1 = basic_conv2d_block(input_shape[0], self.filter * 4)
        self.conv2d_2 = basic_conv2d_block(self.filter * 4, self.filter * 8)
        self.conv2d_3 = basic_conv2d_block(self.filter * 8, self.filter * 10)
        self.conv2d_4 = basic_conv2d_block(self.filter * 10, self.filter * 14)
        self.conv2d_5 = basic_conv2d_block(self.filter * 14, self.filter * 16)
        self.conv2d_6 = basic_conv2d_block(self.filter * 16, self.filter * 17)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(.15)
        self.fc = nn.Linear(272, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_avg_pooling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x = self.max_pooling(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class CNN(nn.Module):
    def __init__(self, input_shape, class_number=2):
        super(CNN, self).__init__()
        self.classes = class_number
        self.input_shape = input_shape
        self.filter = 18
        self.first_avg_pooling = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2d_1 = basic_conv2d_block(input_shape[0], self.filter*2)
        self.conv2d_2 = basic_conv2d_block(self.filter*2, self.filter*6)
        self.conv2d_3 = basic_conv2d_block(self.filter*6, self.filter*10)
        self.conv2d_4 = basic_conv2d_block(self.filter*10, self.filter*12)
        self.conv2d_5 = basic_conv2d_block(self.filter*12, self.filter*16)
        self.conv2d_6 = basic_conv2d_block(self.filter*16, self.filter*18)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(.2)
        self.fc = nn.Linear(324, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_avg_pooling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x = self.max_pooling(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CNNHeatmap(nn.Module):
    def __init__(self, input_shape, class_number=2):
        super(CNNHeatmap, self).__init__()
        self.classes = class_number
        self.input_shape = input_shape
        self.filter = 16

        self.first_avg_pooling = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2d_1 = basic_conv2d_block(input_shape[0], self.filter)
        self.conv2d_2 = basic_conv2d_block(self.filter, self.filter*2)
        self.conv2d_3 = basic_conv2d_block(self.filter*2, self.filter*3)
        self.conv2d_4 = basic_conv2d_block(self.filter*3, self.filter*4)
        self.conv2d_5 = basic_conv2d_block(self.filter*4, self.filter*5)
        self.heatmap = heatmap_block(self.filter*5, self.classes)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(.2)
        self.fc = nn.Linear(3920, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.first_avg_pooling(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.max_pooling(x)

        y = self.heatmap(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x, y


def basic_conv2d_block(input_channel, output_channel):
    sequential = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return sequential


def heatmap_block(input_channel, classes):
    sequential = nn.Sequential(
        nn.Conv2d(input_channel, classes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(classes),
        nn.ReLU()
    )
    return sequential


def info_model(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


class PreProcess(torch.nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()

    def forward(self, x):
        x = x.float()
        x = torch.transpose(x, 1, 3)
        x = torch.div(x, 255)
        return x
