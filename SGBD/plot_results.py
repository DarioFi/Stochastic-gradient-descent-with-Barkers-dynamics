import json
import matplotlib.pyplot as plt

with open("logs.json", "r") as file:
    data = json.load(file)

for obs in data:
    plt.plot(obs["test_losses"], label="Vanilla")
    plt.plot(obs["test_losses_swa"], label="SWA")
    # plt.yscale("log")
    # plt.ylim(bottom=1e-10)
    plt.ylim(bottom=0)
    plt.title(f"{obs['algorithm']} {obs['model']} {obs['corrected']=} {obs['extreme']=}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

