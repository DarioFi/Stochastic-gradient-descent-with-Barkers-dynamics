import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

import numpy as np

np.random.seed(18)


def SG(grad_U, theta, S_n, sample):
    s = 0
    for i in S_n:
        s += grad_U(theta, sample[0][i], sample[1][i], len(sample))
    return s * len(sample) / len(S_n)


def sgld(K, grad_U, n, sample, theta_0, h):
    N = len(sample[0])
    ts = []
    xs = []
    theta = theta_0
    S_n = np.random.choice(N, (K, n), replace=True)
    for k in range(K):
        incr = SG(grad_U, theta, S_n[k], sample)
        xi = np.random.normal(0, h(k))
        theta = theta + h(k) * incr + xi
        # ts.append(theta)
        ts.append(theta[1])
        xs.append(k * n)
    return theta, xs, ts


if __name__ == '__main__':
    def log_prior_normal(t):
        return -t ** 2 / 2


    def grad_log_prior_normal(t):
        return -t


    def log_data_on_theta(x, t):
        return -(x[0] - t * x[1]) ** 2 / 2


    def grad_u(t, x, y, N):
        # x[0] = y
        # x[1] = x
        # Y = theta * x + e
        return (y - np.dot(t, x)) * x - (t - np.array([0, 0])) / N
        # return (x[1] - t * x[0]) * x[0] - (t+10)/N


    DP = 100
    # data = [(np.array([x, y]), 5 * x - 3 * y + np.random.normal(0, 1)) for x in range(DP) for y in range(DP)]
    X = np.array([np.array([x, y]) for x in range(DP) for y in range(DP)])

    a, b = 3, 5

    Y = np.array([a * x + b * y + np.random.normal(0, 1) for x in range(DP) for y in range(DP)])

    data = (X, Y)

    # print(Y)
    # print(X.mean(axis=0))

    # for min_batch in range(5, 400, 25):
    #     tinit = time.time()
    #     res = sgld(100, grad_u, min_batch, data, np.array([0, 0]), lambda k: 10 ** (-4))
    #     print(f"{min_batch}  time elapsed: {time.time() - tinit}  result: {res}")

    h = lambda k: 0.0001 * (k + 1) ** (-.0)

    GRAD_COMPS = 50000
    res, xs, ts = sgld(GRAD_COMPS // 1, grad_u, 1, data, np.array([0, 0]), h)
    plt.plot(xs, ts, label="SGLD 1 min_batch", color="orange")

    # plt.hist(np.array(ts[100:]), bins=50, density=True)
    # plt.grid()
    # plt.show()

    # res, xs, ts = sgld(GRAD_COMPS // 10, grad_u, 10, data, np.array([0, 0]), h)
    # plt.plot(xs, ts, label="SGLD 10 min_batch", color="blue")

    res, xs, ts = sgld(GRAD_COMPS // 25, grad_u, 25, data, np.array([0, 0]), h)
    plt.plot(xs, ts, label="SGLD 100 min_batch", color="green")

    print(res)
    # res, xs, ts = sgld(GRAD_COMPS // (DP ** 2) + 1, grad_u, DP ** 2, data, np.array([0, 0]), h)
    # plt.plot(xs, ts, label="GD", color="red")

    plt.ylim([-3, 8])

    plt.grid()
    plt.title(f"Y = {a}x + {b}z")
    plt.xlabel("Gradients computed")
    plt.ylabel("Theta")
    x = plt.legend()
    plt.show()

    plt.hist(np.array(ts[200:]), bins=50, density=True)
    plt.grid()
    plt.show()
