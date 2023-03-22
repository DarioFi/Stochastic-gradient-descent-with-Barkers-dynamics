from typing import Iterable, Union, Callable, Optional, List, Dict, Any

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
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

    # Step Method
    def step(self, closure: Optional[Callable[[], float]] = ...):
        sigma = .01
        alfa = 100000
        for group in self.param_groups:
            for p in group['params']:  # iterates over layers, i.e. extra iteration on parameters

                z = np.random.normal(sigma, 0.1 * sigma, p.grad.data.shape).astype(np.float32)
                probs = 1 / (1 + torch.exp(p.grad.data * z * alfa))
                # print(probs.mean(), probs.std())
                # plt.hist(probs)
                # plt.show()
                # exit()
                mask = np.ones(z.shape, dtype=np.float32)

                with np.nditer((probs, mask), op_flags=['readwrite']) as it:
                    for pro, mas in it:
                        if np.random.uniform(0,1) < pro:
                            mas[...] = -1
                p.data -= mask * z
                # p.data -= 1e-3 * p.grad.data

        return .0
