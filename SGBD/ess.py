from collections import OrderedDict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

from main import main

import models


def compute_ess(sel_prob, size, EPOCHS, DS, net, save=True):
    print(net.__name__)
    mod, opt = main(True, net, corrected=False, extreme=False, dataset=DS, epochs=EPOCHS, thermolize_start=0,
                    write_logs=False,
                    ensemble_size=size, sel_prob=sel_prob)

    models_lin = []
    for i, x in enumerate(opt.ensemble):
        sd: OrderedDict = x.state_dict()
        ps = [f.cpu().flatten() for f in sd.values()]
        models_lin.append(torch.cat(ps).numpy())

    models_lin = np.array(models_lin)

    bulk = []
    tail = []
    tot = len(models_lin.T)
    stride = tot // 100
    for i, coord in enumerate(models_lin.T):
        if i % stride == 0:
            print(f"Computing ess for {i=}/{tot}  {round(i / tot * 100, 2)}%")
        if i % 10 != 0: continue
        bulk.append(az.ess(coord, method="bulk"))
        tail.append(az.ess(coord, method="tail"))

    if save:
        data = {
            "model": net.__name__,
            # "bulks": bulk,
            "bulk_avg": sum(bulk) / len(bulk),
            "bulk_min": min(bulk),
            "bulk_median": np.median(bulk),
            # "tails": tail,
            "tail_avg": sum(tail) / len(tail),
            "tail_min": min(tail),
            "tail_median": np.median(tail),
            "dataset": DS,
            "epochs": EPOCHS,
            "select_prob": sel_prob,
            "size": size
        }

        with open("ess_logs.json", 'r') as file:
            j = json.load(file)
            j.append(data)
        with open("ess_logs.json", "w") as file:
            json.dump(j, file, indent=4)


if __name__ == '__main__':
    # az.plot_ess(np.ones(1000), kind="evolution")
    # plt.show()
    compute_ess(1, 20, 20, "CIFAR10", models.LargeModel, True)
    compute_ess(1 / 20, 20, 20, "CIFAR10", models.LargeModel, True)
    compute_ess(1, 20, 20, "CIFAR10", models.MnistResNet, True)
    compute_ess(1 / 20, 20, 20, "CIFAR10", models.MnistResNet, True)
