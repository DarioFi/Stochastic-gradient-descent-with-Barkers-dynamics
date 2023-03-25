from typing import Iterable, Union, Callable, Optional, Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Optimizer

from torch import Tensor

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class SGBD(Optimizer):
    # Init Method:
    def __init__(self, params, n_params, device, defaults: Dict[str, Any], corrected=False, extreme=False):
        super().__init__(params, defaults)
        self.extreme = extreme
        self.isp = False
        self.state = dict()
        self.tau = dict()
        self.sigma = .01
        self.probabilities = None
        self.corrected = corrected
        self.n_params = n_params

        self.grad_avg = dict()

        if device == "cuda":
            self.torch_module = torch.cuda
        else:
            self.torch_module = torch
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                self.tau[p] = 0
                self.grad_avg[p] = []

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        beta = .1
        for group in self.param_groups:
            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters

                z = self.torch_module.FloatTensor(p.grad.data.shape).normal_(self.sigma, 0.1 * self.sigma)

                if self.corrected:
                    self.tau[p] = (1 - beta) * self.tau[p] + beta * p.grad.data.std()
                    tau = self.tau[p]
                    m = abs(tau * z) < 1.702
                    # print(m.mean())
                    alfa = self.torch_module.FloatTensor(z.shape).fill_(1)
                    alfa[m] = 1.702 / ((1.702 ** 2 - tau ** 2 * z[m] ** 2) ** .5)
                    # print(alfa.mean(), alfa.std())
                    scale = 10000
                    probs = 1 / (1 + torch.exp(-p.grad.data * scale * z * alfa))
                else:
                    probs = 1 / (1 + torch.exp(-p.grad.data * z * self.n_params))

                # if self.probabilities is None:
                #     self.probabilities = list(probs.flatten())
                #     self.isp = False
                # if len(self.probabilities) > 1e6 and self.isp is False:
                #     plt.hist(self.probabilities, bins=50)
                #     plt.title(f"Probability distribution using param: {0}")
                #     plt.xlim(0, 1)
                #     plt.show()
                #
                # self.isp = True

                # self.probabilities = []
                # if self.isp is False:
                #     self.probabilities.extend(probs.flatten())

                if self.extreme:
                    sampled = (torch.ceil(probs * 2) - 1) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probs

                p.data += (torch.ceil(sampled) * 2 - 1) * z

                self.grad_avg[p].append(p.grad.data.norm())

        return .0
