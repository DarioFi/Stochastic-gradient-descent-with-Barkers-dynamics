from typing import Iterable, Union, Callable, Optional, Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Optimizer

from torch import Tensor

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class SGBD(Optimizer):
    # Init Method:
    def __init__(self, params, defaults: Dict[str, Any]):
        super().__init__(params, defaults)
        self.isp = False
        self.state = dict()
        self.tau = dict()
        self.sigma = .01
        self.probabilities = None
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                self.tau[p] = 0

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        # alfa = 100000
        beta = .1
        sb = .5
        for group in self.param_groups:
            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters

                self.tau[p] = (1 - beta) * self.tau[p] + beta * p.grad.data.std()

                tau = self.tau[p]

                z = np.random.normal(self.sigma, 0.1 * self.sigma, p.grad.data.shape).astype(np.float32)

                alfa = np.ones(z.shape, dtype=np.float32)

                m = abs(tau * z) < 1.702

                # if abs(tau * z[i]) > 1.702:
                #     alfa[i] = 1
                alfa[m] = 1.702 / ((1.702 ** 2 - tau ** 2 * z[m] ** 2) ** .5)

                # print(tau, alfa.mean(), alfa.std(), self.sigma)

                scale = 10000
                probs = 1 / (1 + torch.exp(p.grad.data * scale * z * alfa))
                if self.probabilities is None:
                    self.probabilities = list(probs.flatten())
                    self.isp = False
                else:
                    # print(len(self.probabilities))
                    if len(self.probabilities) > 1e6 and self.isp is False:
                        plt.hist(self.probabilities, bins=50)
                        plt.title(f"Probability distribution using param: {scale}")
                        plt.xlim(0, 1)
                        plt.show()
                        # self.isp = True
                        self.probabilities = []
                    if self.isp is False:
                        self.probabilities.extend(probs.flatten())

                self.sigma = (1 - sb) * self.sigma + sb * (.01 - tau * z.mean())
                # print(probs.mean(), probs.std())
                # plt.hist(probs)
                # plt.show()
                # exit()

                mask = np.ones(z.shape, dtype=np.float32)

                with np.nditer((probs, mask), op_flags=['readwrite']) as it:
                    for pro, mas in it:
                        if np.random.uniform(0, 1) < pro:
                            mas[...] = -1
                p.data -= mask * z
                # p.data -= 1e-3 * p.grad.data

        return .0
