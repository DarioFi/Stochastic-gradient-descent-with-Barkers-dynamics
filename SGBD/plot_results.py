import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open("logs.json", "r") as file:
    data = json.load(file)

allowed_models = ["medium"]
lower_bound_epochs = 25
upper_bound_epochs = math.inf

for obs in data:

    if any(x not in obs["model"].lower() for x in allowed_models):
        continue

    if not (lower_bound_epochs <= obs["epochs"] <= upper_bound_epochs):
        continue

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")

    ax1.plot(obs["train_losses"], label="Train", color="tab:green")
    ax1.plot(obs["test_losses"], label="Test", color="tab:blue")
    ax1.plot(obs["test_losses_ensemble"], label="ensemble test", color="orange")
    ax2.plot(obs["test_accuracies"], label="Vanilla", color="tab:blue")
    ax2.plot(obs["test_accuracies_ensemble"], label="ensemble", color="orange")

    plt.title(f"{obs['algorithm']} {obs['model']}\nCorrected={obs['corrected']} Extreme={obs['extreme']}")
    ax1.legend()

    ax1.set_ylim(0.00001, .5)
    ax2.set_ylim(90, 100)

    # ax1.set_yscale("log")
    # plt.ylim(bottom=1e-10)
    ax1.grid()
    plt.show()
