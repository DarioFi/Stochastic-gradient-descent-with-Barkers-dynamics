"""Collection of utilities for training and defining models"""
import torch
from torch import nn
import torch.nn.functional as f
from torchvision.models import efficientnet_b0


class CnnMedium(nn.Module):
    def __init__(self, use_cifar=False):
        super().__init__()
        if use_cifar:
            self.conv1 = nn.Conv2d(3, 16, 3, 1)
        else:
            self.conv1 = nn.Conv2d(1, 16, 3, 1)

        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if use_cifar:
            self.fc1 = nn.Linear(1568 * 2, 64)
        else:
            self.fc1 = nn.Linear(1152 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output


class DenseModel(nn.Module):
    def __init__(self, cifar=False):
        super().__init__()
        if cifar:
            self.lin = nn.Linear(32 * 32 * 3, 256)
        else:
            self.lin = nn.Linear(28 * 28, 256)

        self.ff = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(.25),
            nn.Linear(256, 10))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin(x)
        x = self.ff(x)
        x = f.log_softmax(x, dim=1)
        return x


class LargeModel(nn.Module):
    def __init__(self, use_cifar=False):
        super().__init__()
        if use_cifar:
            self.conv1 = nn.Conv2d(3, 64, 3, 1)
        else:
            self.conv1 = nn.Conv2d(1, 64, 3, 1)

        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        output = f.log_softmax(x, dim=1)
        return output


class LogisticReg(nn.Module):
    def __init__(self, cifar=False):
        super().__init__()
        if cifar:
            self.lin = nn.Linear(32 * 32 * 3, 10)
        else:
            self.lin = nn.Linear(28 ** 2, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin(x)
        output = f.log_softmax(x, dim=1)
        return output


class TransformerModel(nn.Module):
    def __init__(self, cifar=False, d_model=64, nhead=4, num_layers=2, output_size=10):
        super(TransformerModel, self).__init__()
        if cifar:
            self.embedding = nn.Linear(32 * 32 * 3, d_model)
        else:
            self.embedding = nn.Linear(28 ** 2, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        # x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # x = x.transpose(0, 1)
        x = self.fc(x)
        output = f.log_softmax(x, dim=1)
        return output


from torchvision.models.resnet import ResNet, BasicBlock


class MnistResNet(ResNet):
    def __init__(self, use_cifar=False):
        super(MnistResNet, self).__init__(BasicBlock, [4, 4, 2, 2], num_classes=10)
        if use_cifar is False:
            self.conv1 = torch.nn.Conv2d(1, 64,
                                         kernel_size=(7, 7),
                                         stride=(2, 2),
                                         padding=(3, 3), bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(3, 64,
                                         kernel_size=(7, 7),
                                         stride=(2, 2),
                                         padding=(3, 3), bias=False)

    def forward(self, x):
        x = super().forward(x)
        output = f.log_softmax(x, dim=1)
        return output


class EfficientNetCustom(nn.Module):
    def __init__(self, use_cifar=False):
        super().__init__()
        self.model = efficientnet_b0(num_classes=10)

    def forward(self, x):
        x = self.model.forward(x)
        o = f.log_softmax(x, dim=1)
        return o


Default_Net = CnnMedium


def hot_loader(path, cl):
    def lod(use_cifar):
        model = cl(use_cifar)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    return lod
