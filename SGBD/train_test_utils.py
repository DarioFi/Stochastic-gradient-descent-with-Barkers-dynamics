from torch import nn
from torch.nn import functional as f

loaded_data = None


def train(model, device, train_loader, optimizer, epoch, log_interval=None, log=True, train_loss=None):
    model.train()
    global loaded_data
    if loaded_data is None:
        loaded_data = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loaded_data.append((batch_idx, (data, target)))

    if True:
        for batch_idx, (data, target) in loaded_data:
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = f.cross_entropy(output, target)
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

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
            test_loss += f.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
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
