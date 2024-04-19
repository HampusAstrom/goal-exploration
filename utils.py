import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os

# TODO possibly use jax again?
def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def plot_targeted_goals(goals, coord_names, path):
    assert len(coord_names) == len(goals[1,:])

    fig = plt.figure(figsize=(2*len(coord_names), 2*len(coord_names)))

    for i in range(len(coord_names)**2):
        col = i % (len(coord_names)-1)
        row = i // (len(coord_names)-1)
        if col < row:
            continue # don't plot diagonal
        ax = fig.add_subplot(len(coord_names)-1, len(coord_names)-1, i+1)
        ax.plot(goals[:,col+1], goals[:,row], ".b")
        if row == 0:
            ax.set_xlabel(coord_names[col+1])
        else:
            ax.set_xticks([])
        if col == len(coord_names)-2:
            ax.set_ylabel(coord_names[row])
        else:
            ax.set_yticks([])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"goal_spread"))

    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)

    # sc = Size.Scaled(1)
    # fh = Size.Fixed(3.5)
    # fv = Size.Fixed(2)

    # fig, ax = plt.subplots(2, 2, figsize=(10,6))
    # h = [sc, fh] * 2 + [sc]
    # v = [sc, fv] * 2 + [sc]
    # divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v)
    # for i in range(2):
    #     for j in range(2):
    #         ax[i,j].set_axes_locator(divider.new_locator(nx=2*i+1, ny=2*j+1))
    # for i in ax.flatten():
    #     i.plot(x, y)
    # plt.savefig('f1.pdf')

    # fig, ax = plt.subplots(3, 2, figsize=(10,9))
    # h = [sc, fh] * 2 + [sc]
    # v = [sc, fv] * 3 + [sc]
    # divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v)
    # for i in range(3):
    #     for j in range(2):
    #         ax[i,j].set_axes_locator(divider.new_locator(nx=2*j+1, ny=2*i+1))
    # for i in ax.flatten():
    #     i.plot(x, y)
    # plt.savefig('f2.pdf')

    # fig, ax = plt.subplots(2, 3, figsize=(15,6))
    # h = [sc, fh] * 3 + [sc]
    # v = [sc, fv] * 2 + [sc]
    # divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v)
    # for i in range(2):
    #     for j in range(3):
    #         ax[i,j].set_axes_locator(divider.new_locator(nx=2*j+1, ny=2*i+1))
    # for i in ax.flatten():
    #     i.plot(x, y)
    # plt.savefig('f3.pdf')