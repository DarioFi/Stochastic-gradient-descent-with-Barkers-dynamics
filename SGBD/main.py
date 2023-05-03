from __future__ import print_function

import platform
import time

import numpy as np

import torch
# torch._dynamo.config.verbose=True


import torch.optim as optim
from matplotlib import pyplot as plt

# import torchvision
# from torch import nn
# from torchvision.models import ResNet18_Weights

from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from SGBD.datasets import get_mnist, get_cifar10
from SGBD.utilities import get_kwargs
from optimizer import SGBD
from model import NNet, train, test

torch.manual_seed(2212)


def main(use_sgdb=True, corrected=False, extreme=False, dataset="MNIST", write_logs=True, epochs=4,
         thermolize_start=1, step_size=None, plot_temp=False, select_model=1 / 20, ensemble_size=0):
    if dataset == "MNIST":
        use_cifar10 = False
    elif dataset == "CIFAR10":
        use_cifar10 = True
    else:
        raise Exception("Invalid Dataset")

    train_kwargs, test_kwargs, device = get_kwargs(batch_size=256, test_batch_size=1000)

    model = NNet(use_cifar10).to(device)
    model = model.to(device)

    model_name = str(model.__class__.__name__)  # If done after compile gives wrong name

    if use_cifar10:
        train_loader, test_loader = get_cifar10(train_kwargs, test_kwargs)
        summary(model, (3, 32, 32,))
    else:
        train_loader, test_loader = get_mnist(train_kwargs, test_kwargs)
        summary(model, (1, 28, 28,))

    if "Linux" in platform.platform() and compile_model is True:
        model = torch.compile(model)

    ensemble = None
    if use_sgdb:
        ensemble = [NNet(use_cifar10).to(device) for _ in range(ensemble_size)]
        if "Linux" in platform.platform() and compile_model is True:
            ensemble = [torch.compile(x) for x in ensemble]
        scheduler = None
        print("N parameters:    ", sum(p.numel() for p in model.parameters()))
        optimizer = SGBD(model.parameters(), n_params=sum(p.numel() for p in model.parameters()), device=device,
                         defaults={}, corrected=corrected, extreme=extreme,
                         ensemble=ensemble, step_size=step_size,
                         thermolize_epoch=thermolize_start, epochs=epochs, batch_n=len(train_loader),
                         select_model=select_model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=1, gamma=.7)

    train_loss = []
    losses = []
    losses_ensemble = []
    losses_swa = []
    accuracies = []
    accuracies_swa = []
    accuracies_ensemble = []

    # swa_model = AveragedModel(model)
    # swa_start = thermolize_start
    swa_start = 1000

    for epoch in range(1, epochs + 1):
        start = time.time()
        temp = []
        train(model, device, train_loader, optimizer, epoch, log_interval=25, log=True, train_loss=temp)
        train_loss.append(sum(temp) / len(temp))
        if epoch % 1 == 0:
            print("STD model:   ", end="")
            l, a, le, ae = test(model, device, test_loader, log=True, test_ensemble=ensemble)
            losses.append(l)
            accuracies.append(a)
            if epoch > thermolize_start:
                losses_ensemble.append(le)
                accuracies_ensemble.append(ae)
            else:
                losses_ensemble.append(np.nan)
                accuracies_ensemble.append(np.nan)

        # if epoch >= swa_start:
        #     swa_model.update_parameters(model)
        #     print(f"SWA model:  ", end="")
        #     l, a = test(swa_model, device, test_loader, log=True)
        #     losses_swa.append(l)
        #     accuracies_swa.append(a)
        # else:
        #     losses_swa.append(np.nan)
        #     accuracies_swa.append(np.nan)
        #
        print(f"{epoch=} - elapsed time: {round(time.time() - start, 1)}s\n")

        if scheduler is not None:
            scheduler.step()

    print(f"Optimizer: {optimizer.__class__}")
    print(f"Parameters: {corrected=} - {extreme=}")
    for i, (l, a) in enumerate(zip(losses, accuracies)):
        print(f"Epoch: {i + 1} - Loss: {l} - Accuracy: {a}")

    if write_logs:
        data = {
            "epochs": epochs,
            "corrected": corrected,
            "extreme": extreme,
            "dataset": dataset,
            # "algorithm": str(optimizer.__class__.__name__) + " fixed step",
            "algorithm": str(optimizer.__class__.__name__),
            "model": model_name,
            "test_losses": losses,
            "test_losses_swa": losses_swa,
            "test_losses_ensemble": losses_ensemble,
            "swa_start": swa_start,
            "test_accuracies": accuracies,
            "test_accuracies_swa": accuracies_swa,
            "test_accuracies_ensemble": accuracies_ensemble,
            "train_losses": train_loss,
        }
        import json
        with open("logs.json", 'r') as file:
            j = json.load(file)
            j.append(data)
        with open("logs.json", "w") as file:
            json.dump(j, file, indent=4)

    if plot_temp:
        hist = optimizer.temperature_history

        fig, ax1 = plt.subplots()

        plt.title(f"CNN trained on CIFAR10 with rate = {optimizer.gamma_rate}")
        # plt.title("Temperatures")
        for data in hist.values():
            ax1.plot(data[0], data[1])

        ax2 = ax1.twinx()
        ax1.set_yscale('log')
        ax2.plot(optimizer.gamma_history, label="stepsize")
        ax1.grid()
        ax1.set_ylabel("Temperatures")
        ax1.set_xlabel(f"Steps (epoch = {len(train_loader)})")
        ax2.legend()
        plt.show()
        print(list(len(x) for x in hist.values()))

    return model, optimizer


compile_model = True

EPOCHS = 30
DS = "CIFAR10"
if __name__ == '__main__':
    main(True, corrected=True, extreme=False, dataset=DS, epochs=EPOCHS, thermolize_start=0, write_logs=True, ensemble_size=30)
    main(True, corrected=False, extreme=False, dataset=DS, epochs=EPOCHS, thermolize_start=0, write_logs=True, ensemble_size=30)
    main(False, corrected=False, extreme=False, dataset=DS, epochs=EPOCHS, thermolize_start=0, write_logs=True, ensemble_size=30)

# Prova a fare 10 epoche di SGDB poi 10 epoche di Adam e vedi che succede


# questions:

# Added correction for curse of dimensionality / temperature
# online mean and variance
# fixed step size / using online means
# why extreme is not working
# acceptance rate / accepting probabilities distribution
# hyperparameter tuning
# actual things to write in the thesis
