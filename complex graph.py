import matplotlib.pyplot as plt
from cmath import exp, pi

import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero

# plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(5, 5))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

r = 2
phi = 20

ax.annotate(text="Im", xy=(0, -r - .2), xytext=(-.19, r + .3), arrowprops=dict(arrowstyle="<-", color="gray"),
            color="gray", fontsize=17)

ax.annotate(text="Re", xy=(-r - .2, 0), xytext=(r + .3, -.1), arrowprops=dict(arrowstyle="<-", color="gray"),
            color="gray", fontsize=17)

pc = [r ** (1 / 3) * exp(complex(0, (k * 2 * pi + phi) / 3)) for k in range(3)]
# plt.plot((0, 0), (-r - .1, r + .1), color="lightgray")
# plt.plot((-r - .1, r + .1), (0, 0), color="lightgray")

x = [0, pc[0].real]
y = [0, pc[0].imag]
ax.fill_between(x, 0, y, color="gray")

x = [k for k in np.arange(pc[0].real, r ** (1 / 3), .001)]
y = [(r ** (2 / 3) - k ** 2) ** .5 for k in np.arange(pc[0].real, r ** (1 / 3), .001)]
ax.fill_between(x, 0, y, color="gray")

# for direction in ["xzero", "yzero"]:
#     # adds arrows at the ends of each axis
#     ax.axis[direction].set_axisline_style("-|>")
#
#     # adds X and Y-axis from the origin
#     ax.axis[direction].set_visible(True)
#
for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

c_inner = plt.Circle((0, 0), r ** .3333, color='black', fill=False)
c_outer = plt.Circle((0, 0), r, color='gray', fill=False, linestyle="--")

names = [r"$\sqrt[3]{r}e^{i\varphi/3}$", r"$\sqrt[3]{r}e^{i(\varphi + 2\pi)/3}$",
         r"$\sqrt[3]{r}e^{i(\varphi + 2*2\pi)/3}$"]
posss = [(0, .18), (-.5, .15), (-.5, -.4)]

ran = .5
ax.add_patch(c_inner)
ax.add_patch(c_outer)
ax.set_xlim(-r - ran, r + ran)
ax.set_ylim(-r - ran, r + ran)

# plt.scatter([x.real for x in pc], [x.imag for x in pc])
cols = ["black", "b", "orange"]
for i in range(3):
    plt.scatter(pc[i].real, pc[i].imag, color=cols[i], linewidths=5)
    plt.plot([0, pc[i].real], [0, pc[i].imag], color=cols[i])
    plt.annotate(names[i], (pc[i].real + posss[i][0], pc[i].imag + posss[i][1]), color=cols[i], fontsize=17)

temp = (r * exp(complex(0, phi)))
plt.scatter(temp.real, temp.imag, color="gray", linewidths=5)
plt.plot([0, temp.real], [0, temp.imag], color="gray", linestyle="--")
plt.annotate(r"$re^{i\varphi}$", (temp.real, temp.imag + 0.15), color="gray", fontsize=17)

plt.annotate(r"$\varphi/3$", (r ** (1 / 3) * .5, .1))


plt.plot()
plt.show()



def number_of_ways(sum_to_obtain, minimum_coin_to_use):
    pass