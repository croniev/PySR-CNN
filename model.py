import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.pool1(x2)
        x4 = self.conv2(x3)
        x5 = self.relu2(x4)
        x6 = self.pool2(x5)
        x7 = x6.view(x6.size(0), -1)
        output = self.out(x7)
        return {
            'in': x,
            'out': output,
            'conv1': x1,
            'relu1': x2,
            'pool1': x3,
            'conv2': x4,
            'relu2': x5,
            'pool2': x6,
            'x7': x7
        }
