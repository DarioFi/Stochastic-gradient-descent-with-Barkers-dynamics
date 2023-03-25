from __future__ import print_function

import matplotlib.pyplot as plt
import torch

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from pythorch_custom import SGBD
from model import MNIST_model, train, test


def main(use_mine=True):
    # Training settings
    lr = 1e-4
    log_interval = 30
    batch_size = 256
    test_batch_size = 1000
    epochs = 8

    use_cuda = torch.cuda.is_available()

    # torch.manual_seed(2212)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # print(device)
    model = MNIST_model().to(device)

    if use_mine:
        scheduler = None
        optimizer = SGBD(model.parameters(), n_params=sum(p.numel() for p in model.parameters()), device=device,
                         defaults={}, corrected=False, extreme=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=1, gamma=.7)
        # scheduler = None

    summary(model, (1, 28, 28,))

    losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, log=True)
        l, a = test(model, device, test_loader, log=True)
        losses.append(l)
        accuracies.append(a)
        if scheduler is not None:
            scheduler.step()

    # for key, value in optimizer.grad_avg.items():
    #     plt.plot(value)
    #     plt.title(str(key.shape))
    #     plt.show()
    # optimizer.grad_avg[key] = []

    print(f"Optimizer: {optimizer.__class__}")
    for i, (l, a) in enumerate(zip(losses, accuracies)):
        print(f"Epoch: {i + 1} - Loss: {l} - Accuracy: {a}")


if __name__ == '__main__':
    main(True)
