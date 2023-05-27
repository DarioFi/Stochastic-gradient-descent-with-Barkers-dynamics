import math
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
                 ensemble=None, thermolize_epoch=None, epochs=None, batch_n=None, step_size=None, alfa_target=1 / 4,
                 select_model=1 / 20):
        super().__init__(params, defaults)

        # used for recording data
        self.gamma_history = []
        self.isp = dict()
        self.probabilities: Dict[:List] = dict()

        # parameters
        self.tau = dict()
        self.sigma = .01
        self.beta = .1
        self.corrected = corrected
        self.n_params = n_params
        self.select_model = select_model
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

        # adaptive size correction for temperature
        self.log_temp = dict()
        self.gamma_base = 1
        self.gamma_rate = 0.1
        self.gamma = self.gamma_base
        self.alfa_target = alfa_target
        self.temperature_history = dict()

        self.log_global_stepsize = dict()
        self.exponent_grad = .8

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

                self.log_temp[p] = 1
                self.log_global_stepsize[p] = 0

                self.online_mean[p] = None
                self.online_var[p] = None
                self.online_count[p] = 0

                self.z[p] = self.torch_module.FloatTensor(p.data.shape)

                self.temperature_history[p] = [[], []]

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        self.batch_counter += 1

        # gamma does not depend on the layer
        self.gamma = self.gamma_base / (self.batch_counter ** self.gamma_rate)
        self.gamma_history.append(self.gamma)
        for group in self.param_groups:

            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters
                # if p.grad is None:
                #     continue

                # update online mean and online var with the new gradient
                self.update_online(p)

                self.z[p] = self.z[p].normal_(0, 1)

                sigma = abs(self.online_mean[p]) ** self.exponent_grad
                sigma *= math.exp(self.log_global_stepsize[p])
                self.z[p] *= 0.1 * sigma
                self.z[p] += sigma

                t = math.exp(self.log_temp[p])
                if self.corrected:
                    tau = torch.sqrt(self.online_var[p])
                    m = abs(tau * self.z[p]) < 1.702
                    alfa_c = self.torch_module.FloatTensor(self.z[p].shape).fill_(1)
                    alfa_c[m] = 1.702 / ((1.702 ** 2 - tau[m] ** 2 * self.z[p][m] ** 2) ** .5)

                    probs = 1 / (1 + torch.exp(-t * p.grad.data * self.z[p] * alfa_c))
                else:
                    probs = 1 / (1 + torch.exp(-t * p.grad.data * self.z[p]))

                # region Temperature correction
                alfa = abs(probs - 0.5).mean()
                self.log_temp[p] = self.log_temp[p] - self.gamma * (alfa - self.alfa_target)

                self.log_global_stepsize[p] -= self.gamma/100 * (alfa - self.alfa_target)

                # print(self.log_temp[p], self.log_global_stepsize[p])
                # self.temperature_history[p][0].append(self.batch_counter)
                # self.temperature_history[p][1].append(math.exp(self.log_temp[p]))
                # endregion

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

                if self.extreme:
                    sampled = (1 - probs) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probs

                temp_var = (torch.ceil(sampled) * 2 - 1) * self.z[p]
                p.data = p.data + temp_var

        # region Replace old models in ensemble
        if len(self.ensemble) > 0:
            if self.batch_counter >= self.thermolize_epoch * self.batch_n:
                if random.uniform(0, 1) < self.select_model:
                    model_mod = self.ensemble.get_last()
                    self.ensemble.rotate()

                    state_dict = model_mod.state_dict()

                    for (name, param), x in zip(state_dict.items(), self.param_groups[0]['params']):
                        param.copy_(x)
        # endregion

        return .0

    def update_online(self, p):
        if self.online_mean[p] is None:
            self.online_mean[p] = p.grad.data
            self.online_var[p] = self.torch_module.FloatTensor(p.grad.data.shape).fill_(0)
        else:
            self.online_mean[p] *= (1 - self.beta)
            self.online_mean[p] += self.beta * p.grad.data
            self.online_var[p] *= (1 - self.beta)
            self.online_var[p] += self.beta * (p.grad.data - self.online_mean[p]) ** 2 * self.batch_n


LOG_PROB = False
