from typing import Iterable, Union, Callable, Optional, Dict, Any, List

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
        self.isp = dict()
        self.state = dict()
        self.tau = dict()
        self.sigma = .01
        self.probabilities: Dict[:List] = dict()
        self.corrected = corrected
        self.n_params = n_params ** 1

        self.grad_exp_avg = dict()
        self.z = dict()

        if device == "cuda":
            self.torch_module = torch.cuda
        else:
            self.torch_module = torch
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                self.tau[p] = 0
                self.isp[p] = False
                self.probabilities[p]: List = None
                self.grad_exp_avg[p] = None
                self.z[p] = self.torch_module.FloatTensor(p.data.shape)

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ..., weight_decay=.0000005):
        beta = .1
        for group in self.param_groups:
            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters

                if self.grad_exp_avg[p] is None:
                    self.grad_exp_avg[p] = p.grad.data
                else:
                    self.grad_exp_avg[p] *= (1 - beta)
                    self.grad_exp_avg[p] += beta * p.grad.data

                self.z[p] = self.z[p].normal_(0, 1)
                self.z[p] *= 0.1 * self.grad_exp_avg[p]
                self.z[p] += self.grad_exp_avg[p]
                #
                # z = self.torch_module.FloatTensor(p.grad.data.shape).normal_(self.sigma, 0.1 * self.sigma)

                if self.corrected:
                    self.tau[p] = (1 - beta) * self.tau[p] + beta * p.grad.data.std()
                    tau = self.tau[p]
                    m = abs(tau * self.z[p]) < 1.702
                    # print(m.mean())
                    alfa = self.torch_module.FloatTensor(self.z[p].shape).fill_(1)
                    alfa[m] = 1.702 / ((1.702 ** 2 - tau ** 2 * self.z[p][m] ** 2) ** .5)
                    # print(alfa.mean(), alfa.std())
                    scale = 10000
                    probs = 1 / (1 + torch.exp(-p.grad.data * scale * self.z[p] * alfa))
                else:
                    probs = 1 / (1 + torch.exp(-p.grad.data * self.z[p] * self.n_params))

                if self.probabilities[p] is None:
                    self.probabilities[p] = list(probs.flatten())
                    self.isp[p] = False
                if len(self.probabilities[p]) > 1e6 and self.isp[p] is False:
                    plt.hist(self.probabilities[p], bins=50)
                    plt.title(f"Probability distribution using param: {p.shape}")
                    plt.xlim(0, 1)
                    plt.show()
                    self.isp[p] = True
                if self.isp[p] is False:
                    self.probabilities[p].extend(probs.flatten())

                if self.extreme:
                    sampled = (1 - probs) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probs

                p.data += (torch.ceil(sampled) * 2 - 1) * self.z[p]
                p.data *= (1 - weight_decay)

        return .0
