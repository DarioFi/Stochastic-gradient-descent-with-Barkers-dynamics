import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open("logs.json", "r") as file:
    data = json.load(file)

allowed_models = ["medium", ]
# allowed_algs = ["*"]
allowed_algs = ["sgbd"]
lower_bound_epochs = 19
upper_bound_epochs = 22
corrected = (True, False)
# corrected = (False,)
print(len(data))
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
# fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.28, hspace=None)
i = 0
for obs in data[30:]:

    if not (obs['corrected'] in corrected):
        continue

    if "*" not in allowed_models:
        # print("large" not in obs["model"].lower())
        if any(x not in obs["model"].lower() for x in allowed_models):
            continue

    if not (lower_bound_epochs <= obs["epochs"] <= upper_bound_epochs):
        continue

    if "*" not in allowed_algs:
        if any(x not in obs["algorithm"].lower() for x in allowed_algs):
            continue
    ax1 = ax[i // 2][i % 2]
    i += 1
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")

    ax1.plot(obs["train_losses"], label="Train", color="red")

    ax1.plot(obs["test_losses"], label="Test loss", color="tab:blue")
    # ax1.plot(obs["test_losses_ensemble"], color="tab:green", label="Loss ensemble")
    ax2.plot(obs["test_accuracies"], label="Test accuracy", color="tab:orange")
    # ax2.plot(obs["test_accuracies_ensemble"], color="tab:purple", label="Accuracy ensemble")

    # plt.title(f"{obs['algorithm']}\nCorrected={obs['corrected']} Extreme={obs['extreme']} alpha*={obs['alfa_target']}")
    plt.title(f"Extreme={obs['extreme']} alpha*={obs['alfa_target']}")
    ax1.legend()
    ax2.legend(loc="upper left")

    if i == 0:
        ax1.set_ylim(0.0, 4)
    else:
        ax1.set_ylim(0.0, 3)

    ax2.set_ylim(0, 80)

    # ax1.set_yscale("log")
    # plt.ylim(bottom=1e-10)

    ax1.grid()
plt.show()
