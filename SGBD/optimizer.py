import random
from typing import Iterable, Union, Callable, Optional, Dict, Any, List

from matplotlib import pyplot as plt

from utilities import CircularList

import torch
from torch.optim import Optimizer

from torch import Tensor

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class SGBD(Optimizer):
    # Init Method:
    def __init__(self, params, n_params, device, defaults: Dict[str, Any], corrected=False, extreme=False,
                 ensemble=None, thermolize_epoch=None, epochs=None, batch_n=None, step_size=None):
        super().__init__(params, defaults)

        # used for recording data
        self.isp = dict()
        self.probabilities: Dict[:List] = dict()

        # parameters
        self.tau = dict()
        self.sigma = .01
        self.corrected = corrected
        self.n_params = n_params
        self.select_model = .05
        self.extreme = extreme
        self.thermolize_epoch = thermolize_epoch
        self.epochs = epochs
        self.batch_n = batch_n
        self.batch_counter = 0

        self.step_size = step_size

        # state vars
        self.state = dict()
        self.online_mean = dict()
        self.online_var = dict()
        self.online_count = dict()
        self.z = dict()
        self.ensemble: CircularList = CircularList(ensemble)

        if device.type == "cuda":
            self.torch_module = torch.cuda
        else:
            self.torch_module = torch

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

                # region Online estimations
                if self.online_mean[p] is None:
                    self.online_mean[p] = p.grad.data
                    self.online_var[p] = self.torch_module.FloatTensor(p.grad.data.shape).fill_(0)
                else:
                    self.online_mean[p] *= (1 - beta)
                    self.online_mean[p] += beta * p.grad.data
                    self.online_var[p] *= (1 - beta)
                    self.online_var[p] += beta * (p.grad.data - self.online_mean[p]) ** 2
                # endregion

                self.z[p] = self.z[p].normal_(0, 1)

                if self.step_size is None:
                    self.z[p] *= 0.1 * self.online_mean[p]
                    self.z[p] += self.online_mean[p]
                else:
                    # self.z[p] *= self.step_size / sum(p.shape)
                    # self.z[p] += self.step_size / sum(p.shape)
                    self.z[p] *= self.step_size / self.n_params
                    self.z[p] += self.step_size / self.n_params

                if self.corrected:
                    tau = torch.sqrt(self.online_var[p] * self.n_params)
                    m = abs(tau * self.z[p]) < 1.702
                    alfa = self.torch_module.FloatTensor(self.z[p].shape).fill_(1)
                    alfa[m] = 1.702 / ((1.702 ** 2 - tau[m] ** 2 * self.z[p][m] ** 2) ** .5)

                    probs = 1 / (1 + torch.exp(-p.grad.data * self.z[p] * alfa * self.n_params))
                else:
                    probs = 1 / (1 + torch.exp(-p.grad.data * self.z[p] * self.n_params))

                # self.isp[p] += 1
                #
                # region Plot probabilities
                # print(self.batch_n, self.batch_counter)
                # print(self.batch_counter // self.batch_n)
                # print(self.isp[p])
                if LOG_PROB:
                    if self.batch_counter / self.batch_n >= 1 and self.probabilities[p] is None:
                        self.probabilities[p] = list(probs.flatten())
                        self.isp[p] = True
                    if self.batch_counter // self.batch_n == 2 and self.isp[p] is True:
                        plt.hist(self.probabilities[p], bins=50)
                        plt.title(f"Probability distribution using param: {p.shape}")
                        plt.xlim(0, 1)
                        plt.show()
                        self.isp[p] = False
                    elif self.probabilities[p] is not None and self.isp[p] is True:
                        self.probabilities[p].extend(probs.flatten())
                # endregion

                if self.batch_counter // self.batch_n == 3:
                    return 0

                if self.extreme:
                    sampled = (1 - probs) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probs

                p.data += (torch.ceil(sampled) * 2 - 1) * self.z[p]

        # region Replace old models in ensemble
        if self.batch_counter >= self.thermolize_epoch * self.batch_n:
            if random.uniform(0, 1) < self.select_model:
                model_mod = self.ensemble.get_last()
                self.ensemble.rotate()

                state_dict = model_mod.state_dict()

                for (name, param), x in zip(state_dict.items(), self.param_groups[0]['params']):
                    param.copy_(x)
        # endregion

        return .0


LOG_PROB = False
