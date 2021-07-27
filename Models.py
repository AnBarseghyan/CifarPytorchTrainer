import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = ['MODELS']


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        base = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
        for param in self.base.parameters():
            param.requires_grad = False
        in_features = base.fc.in_features
        self.drop = nn.Dropout(0.2)
        self.final = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        x = self.final(x)
        return x


class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv2d(256, 512, 5, padding=2, stride=5)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv4(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x


MODELS = {

    'model_1': Model_1,

    'model_2': Model_2,

    'model_3': Model_3

}
