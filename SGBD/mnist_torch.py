from __future__ import print_function
import argparse
import time
import torchvision

import matplotlib.pyplot as plt
import torch

import torch.optim as optim
from torch import nn
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from SGBD.datasets import get_MNIST, get_CIFAR10
from pythorch_custom import SGBD
from model import MNIST_model, train, test


def main(use_sgdb=True, corrected=False, extreme=False):
    # Training settings
    log_interval = 25
    batch_size = 256
    test_batch_size = 1000
    epochs = 16

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(2212)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = get_CIFAR10(train_kwargs, test_kwargs)
    # model = MNIST_model(3).to(device)
    model = torchvision.models.resnet18()
    model = nn.Sequential(
        model,
        nn.Linear(1000, 10),
        nn.LogSoftmax(dim=1)
    )
    model = model.to(device)

    # model = MNIST_model(3).to(device)
    if use_sgdb:
        scheduler = None
        optimizer = SGBD(model.parameters(), n_params=sum(p.numel() for p in model.parameters()), device=device,
                         defaults={}, corrected=corrected, extreme=extreme)
    else:
        # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=1, gamma=.7)
        # scheduler = None

    # summary(model, (3, 32, 32,))
    # summary(model, (1, 28, 28,))

    losses = []
    accuracies = []

    # model = torch.compile(model)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train(model, device, train_loader, optimizer, epoch, log_interval, log=False)
        if epoch % 1 == 0:
            l, a = test(model, device, test_loader, log=False)
            losses.append(l)
            accuracies.append(a)
        # print(f"{epoch=} - elapsed time: {round(time.time() - start, 1)}s\n")
        if scheduler is not None:
            scheduler.step()

    # for key, value in optimizer.grad_avg.items():
    #     plt.plot(value)
    #     plt.title(str(key.shape))
    #     plt.show()
    # optimizer.grad_avg[key] = []

    print(f"Optimizer: {optimizer.__class__}")
    print(f"Parameters: {corrected=} - {extreme=}")
    for i, (l, a) in enumerate(zip(losses, accuracies)):
        print(f"Epoch: {i + 1} - Loss: {l} - Accuracy: {a}")


if __name__ == '__main__':
    main(True, corrected=False, extreme=False)
    main(True, corrected=True, extreme=False)
    main(True, corrected=True, extreme=True)
    main(False, corrected=True, extreme=True)
