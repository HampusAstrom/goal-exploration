import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
import csv
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

def plot_targeted_goals(goals, coord_names, path, figname="goal_spread"):
    assert len(coord_names) == len(goals[1,:])
    order = np.linspace(0, 1, len(goals))

    fig = plt.figure(figsize=(2*len(coord_names), 2*len(coord_names)))

    # add region for reachable area
    # TODO

    for i in range(len(coord_names)**2):
        col = i % (len(coord_names)-1)
        row = i // (len(coord_names)-1)
        if col < row:
            continue # don't plot diagonal
        ax = fig.add_subplot(len(coord_names)-1, len(coord_names)-1, i+1)
        # add markers for goal areas
        # TODO hardcoded override for patho MC
        ax.fill_between([0, 0.07], [0.63, 0.63], [0.5, 0.5],
                        alpha=0.5, fc="salmon", ec="red")
        ax.fill_between([-0.07, 0], [-1.6, -1.6], [-1.73, -1.73],
                        alpha=0.5, fc="gold", ec="goldenrod")

        im = ax.scatter(goals[:,col+1], goals[:,row], c=order, s=1)
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
        #ax.set_xticks([])
        #ax.set_xticks([], minor=True)
        #ax.set_yticks([])
        #ax.set_yticks([], minor=True)

    cbar_ax = fig.add_axes([0.05, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticklocation="left")

    #plt.tight_layout()
    plt.savefig(os.path.join(path,figname), bbox_inches="tight")
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

def get_best_x(folders, num2keep, frac_to_eval=0.2, eval_type="eval_logs"):
    if num2keep <= 0:
        num2keep = len(folders)

    means = []
    for folder in folders:
        datas = []
        experiments = get_all_folders(folder)
        for exp in experiments:
            datas = []
            if not os.path.exists(os.path.join(exp, eval_type)):
                break
            data = np.loadtxt(os.path.join(exp, eval_type, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
            cut = int(frac_to_eval*len(data))
            datas.append(data[-cut:])
        mean = np.mean(datas)
        means.append(mean)
    ind = np.argsort(means)[-num2keep:]

    return np.array(folders)[ind]

def collect_datas(experiments, eval_type="eval_logs"):
    values = []
    means = []
    for exp in experiments:
        if not os.path.exists(os.path.join(exp, eval_type)):
            return None, None
        data = np.loadtxt(os.path.join(exp, eval_type, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
        #avg_data = np.convolve(data, [1]*window, 'valid')/window
        # monitor is sequential, so we need to bundle it by number of experiments
        # in each evaluation, that info is found in completed.txt for now
        # TODO replace this with data from evaluations.npz when it collects correctly
        if os.path.isfile(os.path.join(exp, "completed.txt")):
            with open(os.path.join(exp, "completed.txt"),'r') as measure_info:
                reader = csv.reader(measure_info, delimiter=' ')
                m_info = {k: int(v) for [k, v] in reader}

            data = np.reshape(data, (-1,m_info[eval_type]))

            # TODO calculate mean (and return separate data too?)
            mean = np.mean(data, axis=1)

            means.append(mean)
            values.append(data)

    return means, values

def add_subplot(path, window, ax, eval_type="eval_logs", name=None):
    experiments = get_all_folders(path)
    means, values = collect_datas(experiments, eval_type=eval_type)
    if means is None:
        return

    data = np.mean(means, axis=0)
    std = np.std(means, axis=0)
    avg_data = np.convolve(data, [1]*window, 'valid')/window
    avg_std = np.convolve(std, [1]*window, 'valid')/window
    x = np.linspace(0, len(avg_data)*400, len(avg_data)) # TODO this assumes 400 steps between evals
    if name != None:
        ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + name)
    else:
        ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.07,) # alpha=0.03

def plot_individual_experiments(path, window, eval_type="eval_logs", figname=None):
    experiments = get_all_folders(path)
    means, values = collect_datas(experiments, eval_type=eval_type)
    if means is None:
        return

    # prep fig/ax
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    # plot the mean
    data = np.mean(means, axis=0)
    std = np.std(means, axis=0)
    avg_data = np.convolve(data, [1]*window, 'valid')/window
    avg_std = np.convolve(std, [1]*window, 'valid')/window
    x = np.linspace(0, len(avg_data)*400, len(avg_data)) # TODO this assumes 400 steps between evals
    ax.plot(x, avg_data, label="average", zorder=100)
    # if name != None:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + name)
    # else:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.07, zorder=100) # alpha=0.03

    # plot each
    for i, exp in enumerate(means):
        data = np.convolve(exp, [1]*window, 'valid')/window
        x = np.linspace(0, len(data)*400, len(data)) # TODO this assumes 400 steps between evals
        ax.plot(x, data, label=i)

    # legend
    ax.legend(loc='upper left',
            prop={'size': 8}, # 8
            fancybox=True,
            framealpha=0.2)#, bbox_to_anchor=(1, 0.5))

    # Save
    plt.savefig(os.path.join(path,figname), bbox_inches="tight")
    plt.close(fig)


def plot_all_in_folder(dir,
                       coord_names,
                       num2keep=-1,
                       keywords=[],
                       filter=None,
                       name_override=None,
                       eval_type="eval_logs",
                       goal_plots=False):

    # TODO replace with something that adaps to number of configurations
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink']) *
                           cycler('linestyle', ['-', ':', '--', '-.']))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    window = 200

    # get all experiments
    folders = get_all_folders(dir)

    # separate keywords to always keep
    base = get_names_with(folders, keywords)

    # filter rest of selection
    if filter is not None:
        folders = get_names_with(folders, filter)

    # select best of filtered selection
    folders = get_best_x(folders, num2keep, eval_type=eval_type)

    # merge all that are to be plotted
    folders = np.unique(np.concatenate((folders,np.array(base))))

    folders = sorted(folders)

    # make sure rl-base is always last, for readability as it is not in all graphs
    folders_end = []
    for i, str in enumerate(folders):
        if ("base-rl" in str):
            folders_end.append(folders.pop(i))
    folders += folders_end

    for i, folder in enumerate(folders):
        if name_override != None:
            add_subplot(folder, window, ax, eval_type=eval_type, name=name_override[i])
        else:
            add_subplot(folder, window, ax, eval_type=eval_type)

    ax.legend(loc='upper left',
              prop={'size': 8}, # 8
              fancybox=True,
              framealpha=0.2)#, bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("steps")
    if eval_type == "eval_logs":
        ax.set_ylabel("reward")
    elif eval_type == "eval_logsfew" or eval_type == "eval_logsmany":
        ax.set_ylabel("goal success rate")
    else:
        ax.set_ylabel("?")
    #ax.set_ylim(-100, 200)
    fig.tight_layout()
    name = ""
    if num2keep > 0:
        name += f"top{num2keep}_"
    name += eval_type
    if keywords != []:
        name += f"_including_{keywords}"
    plt.savefig(os.path.join(dir, name))

    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

    # plot individal experiments with means
    for folder in folders:
        plot_individual_experiments(folder,
                                    window=window,
                                    eval_type=eval_type,
                                    figname=f"window{window}_" + name)
        plot_individual_experiments(folder,
                                    window=1,
                                    eval_type=eval_type,
                                    figname="unsmoothed_" + name)

    if not goal_plots:
        return
    # plot goals
    for folder in folders:
        experiments = get_all_folders(folder)
        for exp in experiments:
            goal_file = os.path.join(exp, "goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp)
            goal_file = os.path.join(exp, "initial_targeted_goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="initial_targeted_goals")
            goal_file = os.path.join(exp, "reselect_goal_spread")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="reselect_goal_spread")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--top', default=0)
    parser.add_argument('-k', '--keep', nargs='*', default=[])
    parser.add_argument('-f', '--filter', nargs='*', default=[])
    parser.add_argument('-n', '--name_override', nargs='*', default=None)
    parser.add_argument('-g', '--goal_plots', action='store_true')
    args = parser.parse_args()

    #folder, coord_names = "./output/wrapper/SparsePendulumEnv-v1", ["x", "y", "ang. vel."],
    folder, coord_names = "./output/wrapper/PathologicalMountainCar-v1.1", ["xpos", "velocity"],

    #plot_all_in_folder("./output/wrapper/SparsePendulumEnv-v1", coord_names = ["x", "y", "ang. vel."],
    #plot_all_in_folder("./output/wrapper/PathologicalMountainCar-v1.1", coord_names = ["xpos", "velocity"],
    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logs",
                       )
    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logsfew",
                       )
    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logsmany",
                       goal_plots=args.goal_plots,
                       )