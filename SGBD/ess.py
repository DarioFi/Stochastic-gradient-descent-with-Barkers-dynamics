from collections import OrderedDict

import arviz as az
import numpy as np
import torch

from main import main


def sum_corr(data):
    # print(data)
    tot = 1
    for lag in range(1, len(data) - 3):
        # print(lag)
        # print(data[lag:])
        # print(data[:len(data)-lag])
        x = np.corrcoef(data[lag:], data[:len(data) - lag])[0, 1]
        # print(x)
        tot += 2 * (x)
    print(f"{tot=}")
    return len(data) / tot


def compess(sel_mod, size):
    DS = "MNIST"
    EPOCHS = 4
    mod, opt = main(True, corrected=False, extreme=False, dataset=DS, epochs=EPOCHS, thermolize_start=0,
                    write_logs=False,
                    ensemble_size=size, select_model=sel_mod)

    models_lin = []
    for i, x in enumerate(opt.ensemble):
        sd: OrderedDict = x.state_dict()
        ps = [f.flatten() for f in sd.values()]
        models_lin.append(torch.cat(ps).numpy())
        print(models_lin[i].shape)

    models_lin = np.array(models_lin)

    print(models_lin.shape)
    esses = []
    for sm in models_lin.T:
        # print(sm.shape)
        x = az.ess(sm)
        # print(x)
        print(type(x))
        esses.append(x)

    esses = np.array(esses)
    print(f"ESS with {size=} and frequency={int(1 / sel_mod)}")
    print(esses.mean())
    print(esses.std())


# compess(1, 50)
# compess(1 / 10, 50)


size = 500
print(sum_corr(np.array([x for x in range(size)])))

print(sum_corr(np.random.normal(0, 1, size)))
