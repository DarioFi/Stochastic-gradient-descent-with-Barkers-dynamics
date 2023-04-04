import torch
import torch.nn as nn
import torch.nn.functional as f


class MediumModel(nn.Module):
    def __init__(self, use_cifar=False):
        super().__init__()
        if use_cifar:
            self.conv1 = nn.Conv2d(3, 8, 3, 1)
        else:
            self.conv1 = nn.Conv2d(1, 8, 3, 1)

        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if use_cifar:
            self.fc1 = nn.Linear(int(1152 / 28 / 28 * 32 * 32), 32)
        else:
            self.fc1 = nn.Linear(1152, 32)
        self.fc2 = nn.Linear(32, 10)

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
    def __init__(self, channel=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 64, 3, 1)
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


NNet = MediumModel


# model = torchvision.models.resnet18(ResNet18_Weights)
# model = torchvision.models.resnet18()
# model = nn.Sequential(
#     model,
#     nn.Linear(1000, 10),
#     nn.LogSoftmax(dim=1)
# )

def train(model, device, train_loader, optimizer, epoch, log_interval=None, log=True, train_loss=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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


def test(model, device, test_loader, log=True, test_ensemble=None):
    model.eval()
    test_loss = 0
    correct = 0

    tl_ens = 0
    c_ens = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if test_ensemble is not None:
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
