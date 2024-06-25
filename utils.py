import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
from cycler import cycler

# TODO possibly use jax again?
def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def plot_targeted_goals(goals, coord_names, path):
    assert len(coord_names) == len(goals[1,:])
    order = np.linspace(0, 1, len(goals))

    fig = plt.figure(figsize=(2*len(coord_names), 2*len(coord_names)))

    for i in range(len(coord_names)**2):
        col = i % (len(coord_names)-1)
        row = i // (len(coord_names)-1)
        if col < row:
            continue # don't plot diagonal
        ax = fig.add_subplot(len(coord_names)-1, len(coord_names)-1, i+1)
        im = ax.scatter(goals[:,col+1], goals[:,row], c=order)
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

    cbar_ax = fig.add_axes([0, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #plt.tight_layout()
    plt.savefig(os.path.join(path,"goal_spread"))
    plt.close(fig)

def get_all_folders(dir):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

    # for obj in subfolders:
    #     print(obj)
    return sorted(subfolders)

def add_subplot(path, window, ax):
    datas = []
    experiments = get_all_folders(path)
    for exp in experiments:
        data = np.loadtxt(os.path.join(exp, "eval_logs", "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
        #avg_data = np.convolve(data, [1]*window, 'valid')/window
        datas.append(data)
    #avg_data = np.mean(datas, axis=0)
    #avg_std = np.std(datas, axis=0)
    data = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    avg_data = np.convolve(data, [1]*window, 'valid')/window
    avg_std = np.convolve(std, [1]*window, 'valid')/window
    #x = range(len(avg_data))
    x = np.linspace(0, len(avg_data)*100, len(avg_data))
    ax.plot(x, avg_data, label=os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.03,)

def plot_all_in_folder(dir):
    subfolders = get_all_folders(dir)

    # TODO replace with something that adaps to number of configurations
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                           cycler('linestyle', ['-', ':',])))# '--', '-.'])))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    window = 1000

    for folder in subfolders:
        add_subplot(folder, window, ax)

    ax.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("steps")
    ax.set_ylabel("reward")
    #ax.set_ylim(-100, 200)
    fig.tight_layout()
    plt.savefig(os.path.join(dir, "eval_results"))

    #coord_names = ["x", "y", "ang. vel."]
    coord_names = ["xpos", "velocity"]
    for folder in subfolders:
        experiments = get_all_folders(folder)
        for exp in experiments:
            goal_file = os.path.join(exp, "goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')#, skiprows=2, usecols=0)
                plot_targeted_goals(goals, coord_names,exp)
