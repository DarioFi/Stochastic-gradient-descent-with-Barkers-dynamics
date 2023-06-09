import platform
import time

import numpy as np

import torch

from torch import optim
from matplotlib import pyplot as plt

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from train_test_utils import train, test
from SGBD.datasets import get_mnist, get_cifar10
from SGBD.utilities import get_kwargs
from optimizer import SGBD
import models

# Set seeds for reproducibility
torch.manual_seed(2212)
np.random.seed(2212)

threshold_accuracy_train = 66.6 / 100
compile_model = True


def main(use_sgdb, nnet, corrected=False, extreme=False, dataset="MNIST", write_logs=True,
         epochs=4, alfa_target=1 / 4, global_stepsize=1,
         thermolize_start=0, step_size=None, plot_temp=False, sel_prob=1 / 20, ensemble_size=0,
         quit_thresh=False) -> tuple:
    """
    Main function that handles the training of the model. The function is fairly complex and does many things.
    :param use_sgdb: Whether to use SGDB or Adam
    :param nnet: Function that returns the model
    :param corrected: Whether to use the corrected version of SGDB
    :param extreme: Whether to use the extreme version of SGDB
    :param dataset: str Dataset to use. Either MNIST or CIFAR10
    :param write_logs: Whether to write logs to file
    :param epochs: Number of epochs to train
    :param alfa_target: Target alfa value, used by the optimizer
    :param global_stepsize: Global stepsize, used by the optimizer
    :param thermolize_start: Epoch to start collecting models for the ensemble
    :param step_size: If None, ignored, otherwise used to set a fix stepsize in SGDB. Used for testing purposes it is quite bad for training
    :param plot_temp: Whether to plot the temperatures, used to produce plots for the thesis
    :param sel_prob: Selection probability of the ensemble, used by SGDB
    :param ensemble_size: Size of the ensemble, used by SGDB, if 0, no ensemble is used
    :param quit_thresh: Whether to quit training if the accuracy is above threshold_accuracy_train. Used only to collect data about convergence speed

    :return: tuple of (model, optimizer)
    """

    if dataset == "MNIST":
        use_cifar10 = False
    elif dataset == "CIFAR10":
        use_cifar10 = True
    else:
        raise Exception("Invalid Dataset")

    train_kwargs, test_kwargs, device = get_kwargs(batch_size=256, test_batch_size=1000)

    model = nnet(use_cifar10).to(device)
    model = model.to(device)

    model_name = str(model.__class__.__name__)  # If done after compile gives wrong name so it is necessary to save it

    # Load dataset
    if use_cifar10:
        train_loader, test_loader = get_cifar10(train_kwargs, test_kwargs)
        summary(model, (3, 32, 32,))
    else:
        train_loader, test_loader = get_mnist(train_kwargs, test_kwargs)
        summary(model, (1, 28, 28,))

    # Compile model if on Linux (compilation is not supported on Windows)
    if "Linux" in platform.platform() and compile_model is True:
        model = torch.compile(model)

    # Initialize optimizer
    ensemble = None
    scheduler = None

    if use_sgdb:

        # Create ensemble and compile it
        ensemble = [nnet(use_cifar10).to(device) for _ in range(ensemble_size)]
        if "Linux" in platform.platform() and compile_model is True:
            ensemble = [torch.compile(x) for x in ensemble]

        optimizer = SGBD(model.parameters(), device=device,
                         defaults={}, corrected=corrected, extreme=extreme,
                         ensemble=ensemble, step_size=step_size, alfa_target=alfa_target,
                         thermolize_epoch=thermolize_start, batch_size=len(train_loader),
                         select_model=sel_prob, global_stepsize=global_stepsize)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=1, gamma=.7)

    # Initialize lists for logging
    train_loss = []
    losses = []
    losses_ensemble = []
    accuracies = []
    accuracies_ensemble = []

    # Training loop
    for epoch in range(1, epochs + 1):
        start = time.time()

        # temp variables for logging
        temp = []
        temp_corrects = []

        # training step
        print(f"{epoch=}")
        train(model, device, train_loader, optimizer, epoch, log_interval=25, log=True, train_loss=temp,
              corrects=temp_corrects)
        train_loss.append(sum(temp) / len(temp))
        acc_train = sum(temp_corrects)

        # quit training if quit_thresh is True and accuracy is above threshold_accuracy_train. Used for data collection
        print(f"Training accuracy {acc_train=}")
        if acc_train > threshold_accuracy_train and quit_thresh:
            print(f"Hit threshold at epoch {epoch}, quitting")
            return model, optimizer

        # testing step
        print("Testing model:   ", end="")
        l, a, le, ae = test(model, device, test_loader, log=True, test_ensemble=ensemble)
        losses.append(l)
        accuracies.append(a)

        # if epoch > thermolize_start also log ensemble data
        if epoch > thermolize_start:
            losses_ensemble.append(le)
            accuracies_ensemble.append(ae)
        else:
            losses_ensemble.append(np.nan)
            accuracies_ensemble.append(np.nan)

        # log time to have an idea of the length of the training
        print(f"{epoch=} - elapsed time: {round(time.time() - start, 1)}s\n")

        if scheduler is not None:
            scheduler.step()

    # print summary of the training
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
            "algorithm": str(optimizer.__class__.__name__),
            "model": model_name,
            "alfa_target": alfa_target,
            "test_losses": losses,
            "test_losses_ensemble": losses_ensemble,
            "test_accuracies": accuracies,
            "test_accuracies_ensemble": accuracies_ensemble,
            "train_losses": train_loss,
            "stepsize": global_stepsize,
        }
        import json
        with open("logs.json", 'r') as file:
            j = json.load(file)
            j.append(data)
        with open("logs.json", "w") as file:
            json.dump(j, file, indent=4)

    # plot temperatures, used for data collection
    if plot_temp:
        hist = optimizer.temperature_history

        fig, ax1 = plt.subplots()

        plt.title(f"ResNet18 temperatures")
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


if __name__ == '__main__':
    EPOCHS = 15
    DS = "CIFAR10"
    ensemble_size = 0

    nnet = models.LargeModel

    check_time = False

    main(True, nnet, corrected=True, extreme=False, dataset=DS, epochs=EPOCHS, write_logs=True, alfa_target=1 / 4,
         ensemble_size=ensemble_size)
