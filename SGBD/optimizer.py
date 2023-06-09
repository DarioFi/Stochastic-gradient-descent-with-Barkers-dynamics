import math
import random
from typing import Iterable, Union, Callable, Optional, Dict, Any, List

from matplotlib import pyplot as plt

from utilities import CircularList

import torch
from torch.optim import Optimizer

from torch import Tensor
import numpy as np

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class SGBD(Optimizer):
    """SGDB optimizer. This class inherits from Optimizer and can be used to train a pytorch model."""

    def __init__(self, params, device: torch.device, defaults: Dict[str, Any], corrected=False, extreme=False,
                 ensemble: List = None, thermolize_epoch=None, batch_size=None, step_size=None,
                 alfa_target=1 / 4,
                 select_model=1 / 20, global_stepsize=1, plot_corrected_stats=False):
        """
        :type params: _params_t
        :param params: model.parameters(), necessary for the optimizer
        :param device: torch device
        :param defaults: parameters to pass to the Optimizer super class
        :param corrected: Whether to use the corrected version of SGDB
        :param extreme: Whether to use the extreme version of SGDB
        :param ensemble: Ensemble of models, used by SGDB, use None if it's empty
        :param thermolize_epoch: Epoch to start collecting models for the ensemble
        :param batch_size: Batch size, used for data collecting
        :param step_size: If None, ignored, otherwise used to set a fix stepsize in SGDB. Used for testing purposes as it is quite bad for training
        :param alfa_target: Target alfa value for the temperatures
        :param select_model: Selection probability of the ensemble
        :param global_stepsize: Global stepsize, suggested value is 1
        :param plot_corrected_stats: Whether to plot the corrected statistics, used to produce plots for the thesis. If True it will plot and close the training
        """

        self.n_params = sum(p.numel() for p in params)

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

        self.select_model = select_model
        self.extreme = extreme
        self.thermolize_epoch = thermolize_epoch
        self.batch_n = batch_size
        self.batch_counter = 0

        self.step_size = step_size

        self.plot_corrected_stats = plot_corrected_stats
        if plot_corrected_stats:
            size = 10 ** 4
            self.range = (-.0015, 0.0015)
            self.bins = np.linspace(*self.range, size + 1)
            self.histogram_corrected = np.zeros((size,))

            self.bins_alfa = np.linspace(1., 1.00002, 35 + 1)
            self.histogram_corrected_alfa = np.zeros((35,))
            self.seen = 0

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

        self.global_stepsize = global_stepsize

        self.ensemble: CircularList = CircularList(ensemble)

        if device.type == "cuda":
            self.torch_module = torch.cuda
        else:
            self.torch_module = torch

        self.corrected_statistics = {}

        # Some parameters require layer level initialization
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))
                self.tau[p] = 0
                self.isp[p] = False
                self.probabilities[p]: List = None

                self.log_temp[p] = 1
                # self.global_stepsize[p] = 0

                self.online_mean[p] = None
                self.online_var[p] = None
                self.online_count[p] = 0

                self.z[p] = self.torch_module.FloatTensor(p.data.shape)

                self.temperature_history[p] = [[], []]
                self.corrected_statistics[p] = []

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        self.batch_counter += 1

        # gamma does not depend on the layer
        self.gamma = self.gamma_base / (self.batch_counter ** self.gamma_rate)
        self.gamma_history.append(self.gamma)

        for group in self.param_groups:  # iterates over layers, i.e. extra iteration on parameters
            for p in group['params']:

                # update online mean and online var with the new gradient
                self.update_online(p)

                # Draw step proposal
                self.z[p] = self.z[p].normal_(0, 1)
                sigma = self.online_mean[p]
                self.z[p] *= 0.1 * sigma
                self.z[p] += sigma

                # temperature are stored as logs
                t = math.exp(self.log_temp[p])

                # apply correction
                if self.corrected:
                    tau = torch.sqrt(self.online_var[p])
                    m = abs(tau * self.z[p]) < 1.702

                    self.corrected_statistics[p].append(torch.sum(m) / m.numel())
                    alfa_c = self.torch_module.FloatTensor(self.z[p].shape).fill_(1)
                    alfa_c[m] = 1.702 / ((1.702 ** 2 - tau[m] ** 2 * self.z[p][m] ** 2) ** .5)

                    # plot if self.plot_corrected_stats is True, used to produce plots
                    if self.plot_corrected_stats:
                        self.plot_correction_statistics(p, alfa_c)

                    probabilities = 1 / (1 + torch.exp(-t * p.grad.data * self.z[p] * alfa_c))
                else:
                    probabilities = 1 / (1 + torch.exp(-t * p.grad.data * self.z[p]))

                # Temperature update
                alfa = abs(probabilities - 0.5).mean()
                self.log_temp[p] = self.log_temp[p] - self.gamma * (alfa - self.alfa_target)
                if self.log_temp[p] > 30:
                    self.log_temp[p] = 30

                # logs and plots probabilites, used for data collection
                if torch.isnan(self.log_temp[p]) == True:
                    debug = 0
                    return None

                self.temperature_history[p][0].append(self.batch_counter)
                self.temperature_history[p][1].append(math.exp(self.log_temp[p]))

                if LOG_PROB:
                    if self.batch_counter / self.batch_n >= 1 and self.probabilities[p] is None:
                        self.probabilities[p] = list(probabilities.flatten())
                        self.isp[p] = True
                    if self.batch_counter // self.batch_n == 2 and self.isp[p] is True:
                        plt.hist(self.probabilities[p], bins=50)
                        plt.title(f"Probability distribution using param: {p.shape}")
                        plt.xlim(0, 1)
                        plt.show()
                        self.isp[p] = False
                    elif self.probabilities[p] is not None and self.isp[p] is True:
                        self.probabilities[p].extend(probabilities.flatten())

                # computes the direction to move
                if self.extreme:
                    sampled = (1 - probabilities) * 2 - 1
                else:
                    sampled = self.torch_module.FloatTensor(p.grad.data.shape).uniform_() - probabilities
                temp_var = (torch.ceil(sampled) * 2 - 1) * self.z[p]

                # updates the weights
                p.data = p.data + temp_var * self.global_stepsize

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
        """Updates the online mean and online var of the layer"""
        if self.online_mean[p] is None:  # only ran once
            self.online_mean[p] = p.grad.data
            self.online_var[p] = self.torch_module.FloatTensor(p.grad.data.shape).fill_(0)
        else:
            self.online_mean[p] *= (1 - self.beta)
            self.online_mean[p] += self.beta * p.grad.data
            self.online_var[p] *= (1 - self.beta)
            self.online_var[p] += self.beta * (p.grad.data - self.online_mean[p]) ** 2 * self.batch_n

    def plot_correction_statistics(self, p, alfa_c):
        if self.batch_counter > 500:  # 2 epochs
            self.histogram_corrected += np.histogram(self.z[p].cpu().numpy(), bins=self.bins)[0]
            self.seen += self.z[p].numel()

            self.histogram_corrected_alfa += np.histogram(alfa_c.cpu().numpy(), bins=self.bins_alfa)[0]

        if self.batch_counter > 700:
            plt.figure(figsize=(8, 6))
            fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.28, hspace=0.3)

            ax[0].plot(self.bins[:-1], self.histogram_corrected / self.seen)
            ax[0].fill_between(self.bins[:-1], self.histogram_corrected / self.seen,
                               [0] * len(self.histogram_corrected), alpha=.75)

            ax[0].set_ylabel("Frequency")
            ax[0].set_xlabel("z")
            ax[0].grid()
            ax[0].set_title("Distribution of proposed step z")

            # print(alfa_c)
            ax[1].plot(self.bins_alfa[:-1], self.histogram_corrected_alfa / self.seen)
            ax[1].fill_between(self.bins_alfa[:-1], self.histogram_corrected_alfa / self.seen,
                               [0] * len(self.histogram_corrected_alfa), alpha=.75)
            ax[1].set_yscale("log")
            ax[1].grid()
            ax[1].set_ylabel("Frequency (log-scale)")
            ax[1].set_xlabel(r"$\hat\alpha$")
            ax[1].set_title(r"Distribution of correcting factor $\alpha$")
            plt.show()
            exit()


LOG_PROB = False
