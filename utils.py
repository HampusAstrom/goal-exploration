import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
import csv
import argparse
from cycler import cycler
from itertools import combinations
import functools

from typing import List
from scipy.ndimage import gaussian_filter

eval_freq = 0

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

def threshold_gen(thresh):
    def threshold(x, threshold):
        return (x >= threshold)*1.0

    partial = functools.partial(threshold, threshold=thresh)

    return partial


# exponential moving average like tensorboard
def ema_smooth(scalars: List[float], window: float) -> List[float]:  # Weight between 0 and 1
    weight = 1-1/window # adjustment to make it similar in effect
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def window_smooth(data, window):
    return np.convolve(data, [1]*window, 'valid')/window

def gaussian_window_smooth(data, window):
    sigma = window/4 # adjustment to make it similar in effect
    return gaussian_filter(data, sigma=sigma)

smooth = gaussian_window_smooth
# smooth = window_smooth
# smooth = ema_smooth

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
    if len(coord_names) != len(goals[0,:]):
        return
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
        # ax.fill_between([0, 0.07], [0.63, 0.63], [0.5, 0.5],
        #                 alpha=0.5, fc="salmon", ec="red")
        ax.fill_between([0.5, 0.63], [0, 0], [0.07, 0.07],
                        alpha=0.5, fc="salmon", ec="red")
        # ax.fill_between([-0.07, 0], [-1.6, -1.6], [-1.73, -1.73],
        #                 alpha=0.5, fc="gold", ec="goldenrod")
        ax.fill_between([-1.73, -1.6], [-0.07, -0.07], [0, 0],
                        alpha=0.5, fc="gold", ec="goldenrod")

        im = ax.scatter(goals[:,col], goals[:,row+1], c=order, s=1)
        if row == 0:
            ax.set_xlabel(coord_names[col])
        else:
            ax.set_xticks([])
        if col == len(coord_names)-2:
            ax.set_ylabel(coord_names[row+1])
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

def get_all_folders(dir, name=False, full=False):
    if not os.path.exists(dir):
        return []
    if full:
        subfolders = [(f.name, f.path) for f in os.scandir(dir) if f.is_dir()]
    elif name:
        subfolders = [f.name for f in os.scandir(dir) if f.is_dir()]
    else:
        subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]
    return sorted(subfolders, key=lambda entry: entry[0])

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

def means_for_multi_measure(data, n_eval, func=None):
    data = np.reshape(data, (-1,n_eval))

    if func is not None:
        data = func(data)

    # TODO calculate mean (and return separate data too?)
    mean = np.mean(data, axis=1)
    return mean

def collect_datas(experiments, eval_type="eval_logs", func=None):
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

            # data = np.reshape(data, (-1,m_info[eval_type]))

            # if func is not None:
            #     data = func(data)

            # # TODO calculate mean (and return separate data too?)
            # mean = np.mean(data, axis=1)

            mean = means_for_multi_measure(data, m_info[eval_type], func=func)

            means.append(mean)
            values.append(data)

    return means, values

def add_subplot(path, window, ax, eval_type="eval_logs", name=None, func=None):
    experiments = get_all_folders(path)
    means, values = collect_datas(experiments, eval_type=eval_type, func=func)
    if means is None:
        return
    means = np.array(means)
    if means.ndim > 1:
        data = np.mean(means, axis=0)
        std = np.std(means, axis=0)
    else:
        data = means
        std = np.zeros_like(means)
    avg_data = smooth(data, window)
    avg_std = smooth(std, window)
    x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
    if name != None:
        ax.plot(x, avg_data, label=str(len(means))+ "exps " + f" {np.mean(data):.3f} mean " + name)
    else:
        ax.plot(x, avg_data, label=str(len(means))+ "exps " + f" {np.mean(data):.3f} mean " + os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.07,) # alpha=0.03
    return x

def plot_individual_experiments(path, window, eval_type="eval_logs", figname=None, func=None):
    experiments = get_all_folders(path)
    means, values = collect_datas(experiments, eval_type=eval_type, func=func)
    if means is None:
        return

    # prep fig/ax
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1000*px, 1000*px))

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver']) *
                           cycler('linestyle', ['-',':',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    # plot the mean
    means = np.array(means)
    if means.ndim > 1:
        data = np.mean(means, axis=0)
        std = np.std(means, axis=0)
    else:
        data = means
        std = np.zeros_like(means)
    avg_data = smooth(data, window)
    avg_std = smooth(std, window)
    x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
    ax.plot(x, avg_data, label="average", zorder=100)
    # if name != None:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + name)
    # else:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.07, zorder=100) # alpha=0.03

    # plot each
    for i, exp in enumerate(means):
        data = smooth(exp, window)
        x = np.linspace(0, len(data)*eval_freq, len(data))
        ax.plot(x, data, label=i)

    # legend
    ax.legend(loc='upper left',
            prop={'size': 18}, # 8
            fancybox=True,
            framealpha=0.2).set_zorder(101)#, bbox_to_anchor=(1, 0.5))

    # Save
    plt.savefig(os.path.join(path,figname), bbox_inches="tight")
    plt.close(fig)

def plot_each_goal_in_exp(path,
                          window,
                          n_eval = 1,
                          lst=["eval_logs", "train_logs"],
                          put_last = ['[-1.7, -0.02]', '[0.5, 0.02]'],
                          exclude=True,
                          func=None,
                          figname=None):
    # path should be path ending in exp* here
    # check that exp is completed
    if not os.path.isfile(os.path.join(path, "completed.txt")):
        return None, None
    # find all eval folders
    names = get_all_folders(path, name=True)
    if exclude:
        for entry in lst:
            if entry in names:
                names.remove(entry)
    else:
        names = lst

    # prep fig/ax
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1000*px, 1000*px))

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver', 'gold']) *
    #                     cycler('linestyle', ['-', ':', '--', '-.',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver', 'gold']) *
                    cycler('linestyle', [':',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))
    # TODO find out why this isn't applying

    goal_prop = iter(cycler('color', ['k']) *
                    cycler('linestyle', ['--', '-.', (5, (10, 3))]))

    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

    datas = []
    # if exclude is True:
    #     numbers = {s: int(s.replace("eval_logs_", "")) for s in names}
    #     names = sorted(names, key=numbers.__getitem__)
    add_last = []
    new_names = []
    for name in names:
        label = name.replace("eval_logs_","")
        if label in put_last:
            add_last.append(name)
        else:
            new_names.append(name)
    new_names += add_last
    names = new_names

    for name in names:
        if not os.path.exists(os.path.join(path, name)):
            return None, None
        data = np.loadtxt(os.path.join(path, name, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
        if n_eval > 1:
            data = means_for_multi_measure(data, n_eval=n_eval, func=func)
        else:
            if func is not None:
                data = func(data)
        avg_data = smooth(data, window)
        x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
        label = name.replace("eval_logs_","")
        label = label.replace("[","(")
        label = label.replace("]",")")
        if "(-1.7" in label: # hardcoded override
            label += " hard ext. goal"
        if "(0.5" in label: # hardcoded override
            label += " easy ext. goal"
        # if "47" in label: # hardcoded override
        #     label += " (ext. goal)"
        # if "15" in label: # hardcoded override
        #     label += " (ext. goal)"
        if name in add_last:
            ax.plot(x, avg_data, **next(goal_prop), label=label)
        else:
            ax.plot(x, avg_data, label=label, alpha=0.7)
        datas.append(data)

        # try to get and show initial_v data TODO
        if os.path.exists(os.path.join(path, name, "evaluations.npz")):
            evals = np.load(os.path.join(path, name, "evaluations.npz"))
            lst = evals.files
            for item in lst:
                print(f"{name} {item} {evals[item].flatten()}")

    means = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    avg_data = smooth(means, window)
    if len(names) > 1:
        avg_std = smooth(std, window)
        x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
        ax.plot(x, avg_data, label="average", zorder=100, color='k', linestyle='-')
        ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.07, zorder=1, color='k')
    else:
        avg_std = np.zeros_like(avg_data)

    ax.legend(loc='lower center',
              bbox_to_anchor=(0.5, 0.0), # (0.65, 0.1), cliff #
              prop={'size': 11.5}, # 11.5 patho # 14 frozen # 11 cliff
              fancybox=True,
              ncol=4,
              framealpha=0.9).set_zorder(101)
    ax.set_ylabel("goal success rate")
    ax.set_xlabel("steps")
    ax.set_ylim(-0.2, 1.05) # ax.set_ylim(-0.05, 1.05) cliff

    plt.savefig(os.path.join(path,figname), bbox_inches="tight")
    plt.close(fig)

    return avg_data, avg_std

def plot_all_in_folder(dir,
                       coord_names,
                       num2keep=-1,
                       keywords=[],
                       filter=None,
                       name_override=None,
                       eval_type="eval_logs",
                       goal_plots=False,
                       indi_plots=True,
                       cutoff=None,
                       window=50,# 50
                       func=None,
                       symlog_y=False,
                       ):

    # TODO replace with something that adaps to number of configurations
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver']) *
                           cycler('linestyle', ['-',':',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))
    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k']) +
    #                            cycler('linestyle', ['-', ':','--', '-.'])))

    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px)) # (1000*px, 1000*px)

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
            folders_end.append(folders[i])
    folders = [x for x in folders if x not in folders_end]
    folders += folders_end

    for i, folder in enumerate(folders):
        if name_override != None:
            x = add_subplot(folder, window, ax, eval_type=eval_type, name=name_override[i], func=func)
        else:
            x = add_subplot(folder, window, ax, eval_type=eval_type, func=func)

    # oracle = np.ones(x.shape)*0.7
    # ax.plot(x, oracle, label="oracle", zorder=100, color='magenta', linestyle=':')

    if cutoff is not None:
        ax.set_xlim([0, cutoff])

    # if func is None: # cliff
    #     ax.set_ylim(top=-2)

    # HANDMADE OVERRIDE
    # names = ["Intermediate Success", "Novelty",
    #          "Uniform Random", "Baseline"]
    # for i, name in enumerate(names):
    #     ax.lines[i].set_label(name)

    ax.legend(loc='upper left', # loc='upper left', # 'lower right'
              prop={'size': 8}, # 8 # 18
              fancybox=True,
              framealpha=0.2).set_zorder(101)#, bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("steps")
    if eval_type == "eval_logs":
        ax.set_ylabel("reward")
    elif eval_type == "eval_logsfew" or eval_type == "eval_logsmany":
        ax.set_ylabel("goal success rate")
    else:
        ax.set_ylabel("?")

    if func is not None:
        ax.set_ylabel("External goal success rate")

    #ax.set_ylim(-100, 200)
    if symlog_y:
        ax.set_yscale('symlog')
    fig.tight_layout()
    name = ""
    if num2keep > 0:
        name += f"top{num2keep}_"
    name += eval_type
    if keywords != []:
        name += f"_including_{keywords}"
    if func is not None:
        name += "_" + "func"
    plt.savefig(os.path.join(dir, name))

    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

    #plot individal experiments with means
    for folder in folders:
        plot_individual_experiments(folder,
                                    window=window,
                                    eval_type=eval_type,
                                    figname=f"window{window}_" + name,
                                    func=func)
        plot_individual_experiments(folder,
                                    window=1,
                                    eval_type=eval_type,
                                    figname="unsmoothed_" + name,
                                    func=func)

    if not goal_plots:
        return
    # plot goals
    for folder in folders:
        experiments = get_all_folders(folder)
        for exp in experiments:
            goal_file = os.path.join(exp, "goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="goal_spread")
            goal_file = os.path.join(exp, "initial_targeted_goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="initial_targeted_goals")
            goal_file = os.path.join(exp, "reselect_goal_spread")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="reselect_goal_spread")
            goal_file = os.path.join(exp, "successful_goal_spread")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="successful_goal_spread")
            goal_file = os.path.join(exp, "successful_goal_obs_spread")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')
                plot_targeted_goals(goals, coord_names,exp,figname="successful_goal_obs_spread")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--top', default=0)
    parser.add_argument('-k', '--keep', nargs='*', default=[])
    parser.add_argument('-f', '--filter', nargs='*', default=[])
    parser.add_argument('-n', '--name_override', nargs='*', default=None)
    parser.add_argument('-g', '--goal_plots', action='store_true')
    parser.add_argument('-c', '--cutoff', type=int, default=None)
    args = parser.parse_args()

    #folder, coord_names = "./output/wrapper/SparsePendulumEnv-v1", ["x", "y", "ang. vel."],
    folder, coord_names = "./output/wrapper/PathologicalMountainCar-v1.1", ["xpos", "velocity"], # _high_freq_eval
    #folder, coord_names = "./output/wrapper/FrozenLake-v1", ["index"], # TODO replace with grid reshape
    #folder, coord_names = "./output/wrapper/CliffWalking-v0", ["index"]

    #folder, coord_names = "./output/wrapper/PathologicalMountainCar-v1.1_outdated_goal_selection_versions/intermediate_prior_to_0.5_factor_in_target_vs_visits", ["xpos", "velocity"]

    #eval_freq = 100000
    #eval_freq = 50000 #50000 frozen, 20000 for patho MC, (high freq 10000)
    if "FrozenLake" in folder:
        eval_freq = 10000
        thresh = 1
        symlog_y = False
        n_eval = 5
        put_last = ['15']
    elif "Patho" in folder:
        eval_freq = 50000
        thresh = 11
        symlog_y = False
        n_eval = 1
        put_last = ['[-1.7, -0.02]', '[0.6, 0.02]']
    elif "Cliff" in folder:
        eval_freq = 50000
        thresh = -14.0
        symlog_y = True
        n_eval = 1
        put_last = ['47']

    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logs",
                       cutoff=args.cutoff,
                       window=10, #10
                       symlog_y=symlog_y,
                       goal_plots=args.goal_plots,
                       )
    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logs",
                       cutoff=args.cutoff,
                       window=10, #10
                       indi_plots=False,
                       func=threshold_gen(thresh)
                       )

    # plot_all_in_folder(folder,
    #                    coord_names = coord_names,
    #                    num2keep=int(args.top),
    #                    keywords=args.keep,
    #                    filter=args.filter,
    #                    name_override=args.name_override,
    #                    eval_type="eval_logsfew",
    #                    cutoff=args.cutoff,
    #                    )
    # plot_all_in_folder(folder,
    #                    coord_names = coord_names,
    #                    num2keep=int(args.top),
    #                    keywords=args.keep,
    #                    filter=args.filter,
    #                    name_override=args.name_override,
    #                    eval_type="eval_logsmany",
    #                    goal_plots=args.goal_plots,
    #                    cutoff=args.cutoff,
    #                    )
    setups = get_all_folders(folder, full=True)
    avg_datas = []
    avg_stds = []
    names = []
    num_exps = []
    for name, path in setups:
        exps = get_all_folders(path)
        exp_datas = []
        exp_stds = []
        for exp in exps:
            # plot goal successes
            avg_data, avg_std = plot_each_goal_in_exp(exp,
                                                      window=20, # 20
                                                      n_eval=n_eval,
                                                      put_last=put_last,
                                                      figname="goal_success_rate")
            # plot training success rate
            plot_each_goal_in_exp(exp,
                                  100,
                                  n_eval=1,
                                  lst=["train_logs"],
                                  exclude=False,
                                  func=threshold_gen(1), # TODO replace with non pmc hardcoded
                                  figname="train_success_rate")
            if avg_data is None or avg_data.shape == ():
                continue
            exp_datas.append(avg_data)
            exp_stds.append(avg_std)
        if len(exps) > 1:
            exp_datas = np.mean(exp_datas, 0)
            exp_stds = np.mean(exp_stds, 0)
        else:
            exp_datas = np.array(exp_datas)
            exp_stds = np.array(exp_stds)
        if exp_datas.shape != ():
            avg_datas.append(exp_datas.flatten())
            avg_stds.append(exp_stds.flatten())
            names.append(name)
            num_exps.append(len(exps))

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k']) +
    #                         cycler('linestyle', ['-', ':','--', '-.'])))
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver']) *
                        cycler('linestyle', ['-',':',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

    for name, data, std, num_exp in zip(names, avg_datas, avg_stds, num_exps):
        if data.size == 0:
            break
        x = np.linspace(0, len(data)*eval_freq, len(data))
        ax.plot(x, data, label=str(num_exp)+ "exps " + f" {np.mean(data):.3f} mean " + name, zorder=2)
        ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)

    # names = ["Intermediate Success", "Novelty",
    #          "Uniform Random",]
    # for i, name in enumerate(names):
    #     ax.lines[i].set_label(name)

    ax.legend(loc='lower left', # 'upper left' # 'lower right'
              prop={'size': 8}, # 8 # 18
              fancybox=True,
              framealpha=0.8).set_zorder(101)#, bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("goal success rate")
    ax.set_xlabel("steps")

    plt.savefig(os.path.join(folder,"average_goal_success_rate"), bbox_inches="tight")
    plt.close(fig)