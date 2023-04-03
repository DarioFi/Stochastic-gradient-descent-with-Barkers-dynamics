import random
from typing import Iterable, Union, Callable, Optional, Dict, Any, List

from utilities import CircularList

import torch
from torch.optim import Optimizer

from torch import Tensor

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class SGBD(Optimizer):
    # Init Method:
    def __init__(self, params, n_params, device, defaults: Dict[str, Any], corrected=False, extreme=False,
                 ensemble=None, thermolize_epoch=None, epochs=None, batch_n=None):
        super().__init__(params, defaults)
        self.select_model = .05
        self.extreme = extreme
        self.isp = dict()
        self.state = dict()
        self.tau = dict()
        self.sigma = .01
        self.probabilities: Dict[:List] = dict()
        self.corrected = corrected
        self.n_params = n_params ** 1

        self.online_mean = dict()
        self.online_var = dict()
        self.online_count = dict()
        self.z = dict()

        # print(device.type)
        if device.type == "cuda":
            self.torch_module = torch.cuda
        else:
            self.torch_module = torch

        # print(self.torch_module)

        self.ensemble: CircularList = CircularList(ensemble)
        self.thermolize_epoch = thermolize_epoch
        self.epochs = epochs
        self.batch_n = batch_n
        self.batch_counter = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                self.tau[p] = 0
                self.isp[p] = False
                self.probabilities[p]: List = None

                self.online_mean[p] = None
                self.online_var[p] = None
                self.online_count[p] = 0

                self.z[p] = self.torch_module.FloatTensor(p.data.shape)

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        beta = .1

        self.batch_counter += 1

        for group in self.param_groups:
            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters

                if self.online_mean[p] is None:
                    self.online_mean[p] = p.grad.data
                    self.online_var[p] = self.torch_module.FloatTensor(p.grad.data.shape).fill_(0)
                else:
                    self.online_mean[p] *= (1 - beta)
                    self.online_mean[p] += beta * p.grad.data
                    self.online_var[p] = beta * self.online_var[p] + (1 - beta) * (
                            p.grad.data - self.online_mean[p]) ** 2

                self.z[p] = self.z[p].normal_(0, 1)
                self.z[p] *= 0.1 * self.online_mean[p]
                self.z[p] += self.online_mean[p]

                # z = self.torch_module.FloatTensor(p.grad.data.shape).normal_(self.sigma, 0.1 * self.sigma)

                if self.corrected:
                    tau = torch.sqrt(self.online_var[p] * self.n_params)
                    m = abs(tau * self.z[p]) < 1.702
                    alfa = self.torch_module.FloatTensor(self.z[p].shape).fill_(1)
                    alfa[m] = 1.702 / ((1.702 ** 2 - tau[m] ** 2 * self.z[p][m] ** 2) ** .5)

                    probs = 1 / (1 + torch.exp(-p.grad.data * self.z[p] * alfa * self.n_params))
                else:
                    probs = 1 / (1 + torch.exp(-p.grad.data * self.z[p] * self.n_params))

                self.isp[p] += 1

                # if self.isp[p] > 10 and self.probabilities[p] is None:
                #     self.probabilities[p] = list(probs.flatten())
                # if self.isp[p] == 20:
                #     plt.hist(self.probabilities[p], bins=50)
                #     plt.title(f"Probability distribution using param: {p.shape}")
                #     plt.xlim(0, 1)
                #     plt.show()
                # elif self.probabilities[p] is not None:
                #     self.probabilities[p].extend(probs.flatten())

                if self.extreme:
                    sampled = (1 - probs) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probs

                p.data += (torch.ceil(sampled) * 2 - 1) * self.z[p]

        if self.batch_counter >= self.thermolize_epoch * self.batch_n:
            if random.uniform(0, 1) < self.select_model:
                model_mod = self.ensemble.get_last()
                self.ensemble.rotate()

                state_dict = model_mod.state_dict()

                for (name, param), x in zip(state_dict.items(), self.param_groups[0]['params']):
                    param.copy_(x)

        return .0