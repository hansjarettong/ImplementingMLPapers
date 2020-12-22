import torch
import torch.nn as nn
import torch.nn.functional as F

class _AlexNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, stride = 4, padding = 2)
        self.conv2 = nn.Conv2d(48, 128, 5, padding = 2)
        self.res_norm = nn.LocalResponseNorm(5, k = 2)
        self.max_pool = nn.MaxPool2d(3, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_norm(x)
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        return x


class _AlexNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_norm = nn.LocalResponseNorm(5, k = 2)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(256, 192, 3, padding = 1)
        self.conv4 = nn.Conv2d(192, 192, 3, padding = 1)
        self.conv5 = nn.Conv2d(192, 128, 3, padding = 1)
        
        
    def forward(self, x):
        x = self.res_norm(x)
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, 2)
        self.alexnet1a = _AlexNet1()
        self.alexnet1b = _AlexNet1()
        self.alexnet2a = _AlexNet2()
        self.alexnet2b = _AlexNet2()
        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        
    def forward(self, x):
        x1 = self.alexnet1a(x)
        x2 = self.alexnet1b(x)
        x = torch.cat([x1,x2], dim = 1)
        x1 = self.alexnet2a(x)
        x2 = self.alexnet2b(x)
        x = torch.cat([x1,x2], dim = 1)
        x = self.max_pool(x)
        x = x.flatten(start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x