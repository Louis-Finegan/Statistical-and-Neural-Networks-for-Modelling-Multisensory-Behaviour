"""Network architectures used in the undergraduate experiments."""
import torch
from torch import nn

class ANN1(nn.Module):

    def __init__(self):
        super(ANN1, self).__init__()
        self.hidden = nn.Linear(2, 1, bias=False)
        self.output = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

class ANN2(nn.Module):

    def __init__(self):
        super(ANN2, self).__init__()
        self.hidden = nn.Linear(2, 4, bias=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

class ANN3(nn.Module):

    def __init__(self):
        super(ANN3, self).__init__()
        self.hidden = nn.Linear(2, 10, bias=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1, bias=True)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

class ANN_MAP3(nn.Module):

    def __init__(self):
        super(ANN_MAP3, self).__init__()
        self.hidden1 = nn.Linear(2, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 3)
        self.elu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.elu(self.hidden1(x))
        x = self.elu(self.hidden2(x))
        outputs = self.output(x)
        return torch.cat((outputs[:, :2], self.sigmoid(outputs[:, 2:3])), dim=1)
