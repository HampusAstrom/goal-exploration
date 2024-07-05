import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
import argparse
from cycler import cycler
from itertools import combinations

# TODO possibly use jax again?
def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))
    # TODO use np.log1p

def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def norm_vec(x):
    std = np.std(x)
    if std == 0:
        std = 1
    return (x-np.mean(x))/std

# num is number of elements to fill, should be between 1 and len(weights)
def weight_combinations(weights, num):
    weights = np.array(weights)
    indices = list(range(len(weights)))
    combs = combinations(indices, num)
    configurations = []
    for comb in combs:
        comb = np.array(comb)
        conf = np.zeros(weights.shape)
        conf[comb] = weights[comb]
        # for readability of returned list
        lst = []
        is_int = np.mod(conf, 1) == 0
        for i in range(is_int.size):
            if is_int[i]:
                lst.append(int(conf[i]))
            else:
                lst.append(float(conf[i]))
        configurations.append(lst)
    return configurations

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
    if not os.path.exists(dir):
        return []
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    return sorted(subfolders)

def get_names_with(strarray, substrngs):
    ret = []
    for str in strarray:
        if any(s in str for s in substrngs):
            ret.append(str)
    return ret

def get_best_x_and_keywords(path, num2keep, keywords=[], frac_to_eval=0.2):
    folders = get_all_folders(path)
    base = get_names_with(folders, keywords)

    if num2keep <= 0:
        num2keep = len(folders)

    means = []
    for folder in folders:
        datas = []
        experiments = get_all_folders(folder)
        for exp in experiments:
            datas = []
            data = np.loadtxt(os.path.join(exp, "eval_logs", "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
            cut = int(frac_to_eval*len(data))
            datas.append(data[-cut:])
        mean = np.mean(datas)
        means.append(mean)
    ind = np.argsort(means)[-num2keep:]
    print(ind)

    return np.unique(np.concatenate((np.array(folders)[ind],np.array(base))))

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
    ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.03,)

def plot_all_in_folder(dir, num2keep, keywords=["baseline"]):
    folders = get_all_folders(dir)

    # TODO replace with something that adaps to number of configurations
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'c', 'y', 'm']) *
                           cycler('linestyle', ['-', ':', '--', '-.',]))) # '-.', (5, (10, 3))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    window = 1000

    # override to only plot x best + baselines
    folders = get_best_x_and_keywords(dir, num2keep, keywords=keywords)

    folders = sorted(folders)

    for folder in folders:
        add_subplot(folder, window, ax)

    ax.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("steps")
    ax.set_ylabel("reward")
    #ax.set_ylim(-100, 200)
    fig.tight_layout()
    name = ""
    if num2keep > 0:
        name += f"top{num2keep}_"
    name += "eval_results"
    if num2keep > 0 and keywords != []:
        name += f"_including_{keywords}"
    plt.savefig(os.path.join(dir, name))

    #coord_names = ["x", "y", "ang. vel."]
    coord_names = ["xpos", "velocity"]
    for folder in folders:
        experiments = get_all_folders(folder)
        for exp in experiments:
            goal_file = os.path.join(exp, "goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')#, skiprows=2, usecols=0)
                plot_targeted_goals(goals, coord_names,exp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--top', default=0)
    parser.add_argument('-k', '--keep', nargs='*', default=[])
    args = parser.parse_args()

    #plot_all_in_folder("./output/wrapper/SparsePendulumEnv-v1") #
    plot_all_in_folder("./output/wrapper/PathologicalMountainCar-v1.1",
                       int(args.top),
                       keywords=args.keep)