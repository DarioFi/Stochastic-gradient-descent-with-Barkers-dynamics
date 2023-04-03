import copy
import math
import matplotlib.pyplot as plt

import numpy as np

errs = 0
np.random.seed(2212)


def CSGDBD(batch_size, n_iter, grad, theta, sample, beta, dims):
    bs = batch_size
    thetas = [theta]
    tau = 1
    for t in range(n_iter):
        s_n = np.random.choice(sample, bs, replace=True)

        temp = 0
        comp_grad = grad(s_n, theta)
        for s in s_n:
            temp += (grad([s], theta) - comp_grad) ** 2

        tau = (1 - beta) * tau + beta * np.sqrt(
            bs / (bs - 1) * temp
        )

        sigma = 1.702 / 1.233 / tau
        z = np.random.normal(sigma, 0.1 * sigma, dims)
        for i in range(dims):
            alfa = np.ones(dims)
            if abs(tau[i] * z[i]) > 1.702:
                global errs
                errs += 1
                alfa[i] = 1
            else:
                alfa[i] = 1.702 / (math.sqrt(1.702 ** 2 - tau[i] ** 2 * z[i] ** 2))

            if -z[i] * alfa[i] * comp_grad[i] > 100:
                p = 0
            else:
                p = 1 / (1 + math.exp(-z[i] * alfa[i] * comp_grad[i]))

            if np.random.uniform(0, 1) > p:
                b = -1
            else:
                b = 1
            theta[i] = theta[i] - z[i] * b
        thetas.append(copy.copy(theta))

    return thetas


if __name__ == '__main__':
    def gradient(lista, w):
        # poly = w ** 4 - 2 * w ** 3 - w + 2
        return 1 / len(lista) * np.array([w[0] ** 1 - 1, 5 * w[1] * 3 + 2])


    its = 100000
    res = CSGDBD(4, its, gradient, [0, 0], [None for x in range(100)], .1, 2)[1000:]
    print(errs / its * 100, "%")
    plt.hist([res[n][0] for n in range(len(res))], bins=100, fc=(0, 0, 1, 0.7))
    plt.hist([res[n][1] for n in range(len(res))], bins=100, fc=(1, 0, 0, 0.7))
    plt.show()
