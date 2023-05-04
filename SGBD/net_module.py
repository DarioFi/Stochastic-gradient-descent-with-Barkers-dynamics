import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torchvision.models import ResNet18_Weights


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


Default_Net = LogisticReg

loaded_data = None


def train(model, device, train_loader, optimizer, epoch, log_interval=None, log=True, train_loss=None):
    model.train()
    global loaded_data
    if loaded_data is None:
        loaded_data = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loaded_data.append((batch_idx, (data, target)))

    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in loaded_data:
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if train_loss is not None:
            train_loss.append(loss.item())
        if batch_idx % log_interval == 0 and log:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


loaded_test = None


def test(model, device, test_loader, log=True, test_ensemble=None):
    model.eval()
    test_loss = 0
    correct = 0

    tl_ens = 0
    c_ens = 0
    global loaded_test
    if loaded_test is None:
        loaded_test = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            loaded_test.append((data, target))
    # with torch.no_grad():
    for _ in range(1):

        for data, target in loaded_test:
            # data, target = data.to(device), target.to(device)
            # model.eval()
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if False and test_ensemble is not None and len(test_ensemble) > 0:
                data, target = data.to(device), target.to(device)
                output = None
                for mod in test_ensemble:
                    if output is None:
                        output = mod(data)
                    else:
                        output += mod(data)
                output /= len(test_ensemble)
                tl_ens += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                c_ens += pred.eq(target.view_as(pred)).sum().item()

        # tl_ens /= len(test_ensemble)
        # c_ens /= len(test_ensemble)

    test_loss /= len(test_loader.dataset)
    tl_ens /= len(test_loader.dataset)

    if log:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        print('Ensemble test: Average loss: {:.4f}, Accuracy: {:.1f}/{} ({:.0f}%)'.format(
            tl_ens, c_ens, len(test_loader.dataset),
            100. * c_ens / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset), tl_ens, 100. * c_ens / len(test_loader.dataset)
