import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open("logs.json", "r") as file:
    data = json.load(file)

allowed_models = ["medium"]
allowed_algs = ["sgbd"]
# allowed_algs = ["adam"]
lower_bound_epochs = 15
upper_bound_epochs = math.inf
# corrected = (True, False)
corrected = (True,)

for obs in data:

    if not obs['corrected'] in corrected:
        continue

    if any(x not in obs["model"].lower() for x in allowed_models):
        continue
    if not (lower_bound_epochs <= obs["epochs"] <= upper_bound_epochs):
        continue

    if "*" not in allowed_algs:
        if any(x not in obs["algorithm"].lower() for x in allowed_algs):
            continue

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")

    ax1.plot(obs["train_losses"], label="Train", color="red")

    ax1.plot(obs["test_losses"], label="Test loss", color="tab:blue")
    ax1.plot(obs["test_losses_ensemble"], color="tab:green", label="Loss ensemble")
    ax2.plot(obs["test_accuracies"], label="Test accuracy", color="tab:orange")
    ax2.plot(obs["test_accuracies_ensemble"], color="tab:orange")

    plt.title(f"{obs['algorithm']} {obs['model']}\nCorrected={obs['corrected']} Extreme={obs['extreme']}")
    ax1.legend()
    ax2.legend(loc="upper left")

    ax1.set_ylim(0, 3)
    ax2.set_ylim(0, 100)

    # ax1.set_yscale("log")
    # plt.ylim(bottom=1e-10)

    ax1.grid()
    plt.show()
