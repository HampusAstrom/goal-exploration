import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
import csv
import argparse
from cycler import cycler
from itertools import combinations
import functools
from copy import deepcopy
from mergedeep import merge as deepmerge
from mergedeep import Strategy as MergeStrat
from deepdiff import DeepDiff, Delta
from deepdiff.operator import BaseOperatorPlus
from pprint import pprint

from typing import List, Any, Callable, Dict, Optional, Union
from scipy.ndimage import gaussian_filter
import scipy.stats as stats

eval_freq = 0

# Helper functions for derivative logs

def meta_eval_reward_quick_and_no_v_explode(values,
                                            rew_window=10,
                                            rew_thresh=-110,
                                            init_v_thresh=10,
                                            rew_range=[-1000, -100], # rew_min, rew_max
                                            weights=[1, 1, 1], # w_a, w_b, w_c
                                            ):
    # after zip, values should be:
    # values[0] should be last_mean_reward
    # values[1] should be last_mean_initial_values
    # values[2] should be step
    # values[3] should be max_steps
    # assume window is full run
    # equation:                                             (lower values are better)
    # sfrac = step/max_step (at end of window) [0,1]        (each component 0-1)
    # when rew first over thresh, a=sfrac, else a=1.5       (get over thresh fast)
    # if initial_values over init_v_thresh b=1, else b=0    (don't explode Q)
    # rew_frac = (reward-rew_min)/(rew_max-rew_min)         (norm to possible rewards)
    # (highest windowed) reward gives c = 1-rew_frac
    # (for c we need to reward high reward, esp. if rew thresh triggered, and low should be better)
    # meta_val = a*w_a + b*w_b + c*w_c
    max_ind = len(values)
    values = np.array(values)
    rews = values[:, 0]
    max_init_v = np.max(values[:, 1])
    steps = values[:, 2]
    max_step = np.max(values[:, 3])
    sfrac = steps/max_step
    win_rews = window_smooth(rews, rew_window) # remember to adjust index of this by window
    where_over = win_rews > rew_thresh
    if where_over.any():
        index = np.argmax(where_over) + rew_window-1 # window of len 1 should not move index
        index = np.min([index, max_ind])
        index = np.min([0, index])
        a = sfrac[index]
    else:
        a = 1.5
    b = 1 if max_init_v > init_v_thresh else 0
    rew_frac = (np.max(win_rews)-rew_range[0])/(rew_range[1]-rew_range[0])
    c = 1-rew_frac

    # TODO make some way to log all the intermediate values here, to see that they are reasonable
    # at least a, b, c
    # and maybe also don't do the full calculation each eval, only the needed parts...

    return a*weights[0] + b*weights[1] + c*weights[2]


def if_v_under_vmax_steps_else_steps_plus_maxsteps(v, vmax, steps, maxsteps):
    if v < vmax:
        return steps
    else:
        return steps + maxsteps

def func_per_val(funcs: List[Callable] = None):
    def ret_func(vals: List[List[Any]]):
        zipped_vals = zip(*vals)
        ret = []
        for val, func in zip(zipped_vals, funcs):
            ret.append(func(val))

    return ret_func

# take mean of whatever is in vals
# check if mean has passed threshold
# over True means if its over, otherwise under
def check_threshold(vals,
                    thresh,
                    over = True,
                    ):
    # whatever we get, take mean in all dimensions
    mean_val = np.mean(vals)
    if over and mean_val > thresh:
        return True
    elif not over and mean_val < thresh:
        return True
    else:
        return False

# end of helper functions for derivative logs

def replace_type_in_dict_from(dct1: dict, dct2: dict, type=list, sort=True):
    if sort: # assure sorted dict while we are looping deeply
        dct1 = dict(sorted(dct1.items()))
    for key, val in dct1.items():
        if isinstance(val, type):
            dct1[key] = dct2[key]
        if isinstance(val, dict) and key in dct2:
            dct1[key] = replace_type_in_dict_from(dct1[key], dct2[key], type)
    return dct1

def first_unique(str1, str2, at_least=3):
    # return first unique strings from start of both strings, assumes existance
    # though at_least # chars, skipped for now
    index = next((i for i, (char1, char2) in enumerate(zip(str1, str2)) if char1 != char2), None)
    if index is None:
        index = min(len(str1), len(str2))
    uq_int1 = min(index, len(str1))
    uq_int2 = min(index, len(str2))
    index = max(index+1, at_least)
    index1 = min(index, len(str1))
    index2 = min(index, len(str2))
    return str1[:index1], str2[:index2], uq_int1, uq_int2

def find_short_strs(str_lst: list[str], sep="_"):
    if len(str_lst) < 1: # TODO this might be wrong
        return ""
    str_lst = sorted(str_lst)
    # find a short unique representation of each string in list
    # first letters if 3 or less
    # if multiple words divided by "_" first letter of each (or some of)
    # first letter +_+ first different part after a "_"?

    # first group by shared start of strings
    prev = str_lst[0]
    begin = {prev: ""}
    ind_for_uniq = {prev: 0}
    for s in str_lst[1:]:
        beg1, beg2, ind1, ind2 = first_unique(prev, s)
        if ind1 > ind_for_uniq[prev]:
            begin[prev] = beg1
            ind_for_uniq[prev] = ind1
        begin[s] = beg2
        ind_for_uniq[s] = ind2
        prev = s

    all_options = {key: [val] for key, val in begin.items()}

    # find reps that come from splitting on "_"
    # there is no guarantee these are unique
    splits = {}
    for s in str_lst:
        if sep in s:
            parts = s.split(sep=sep)
            splits[s] = "".join([string[0] for string in parts])
            all_options[s].append(splits[s])

    # first letter + first x letters of unique part after a "_"
    start_plus_unique = {}
    for s in str_lst:
        if ind_for_uniq[s] < len(s):
            ind = max(ind_for_uniq[s]-2, s.find("_")+1)
            end = s[ind:]
            mx = min(len(end), 3)
            start_plus_unique[s] = s[0] + sep + end[:mx]
            all_options[s].append(start_plus_unique[s])

    final = begin
    # select reps, and make sure the different types above didn't create the
    # same string
    # go through all options, useing begin as default (as it is only one
    # guaranteed to be unique and sorted), find those with worst options
    # and give them replacements first, with target being 3 chars long as ideal
    begin_by_len = dict(sorted(begin.items(), key=lambda x: -len(x[1])))
    for key, val in begin_by_len.items():
        weight_for_obts = np.array([abs(len(val)-3) for val in all_options[key]])
        if (weight_for_obts < abs(len(val)-3)).any():
            best_ind = np.argmin(weight_for_obts)
            best_opt = all_options[key][best_ind]
            if best_opt not in final.values():
                final[key] = best_opt

    return final

def obj2shortstr(obj: Any, info: Any = None):
    # make short string of any object
    # use shared methods for string like things, like from class name
    # remove spaces in lists (and turn [128, 128] into 2x128 when possible)
    # make several options and use shortest one
    # numbers should go from 100.1 -> 1e2, at least when just number (not in list)
    if isinstance(obj, dict):
        return "{" + to_short_string(obj, info) + "}"
    elif isinstance(obj, str):
        return obj[:min(len(obj), 3)]
    elif isinstance(obj, bool):
        if obj == True:
            return ""
        else:
            return "F"
    elif isinstance(obj, (float, int)):
        abs_obj = np.abs(obj)
        if (abs_obj <= 0.01 or abs_obj >= 999):
            string = np.format_float_scientific(obj,
                                                trim="-",
                                                exp_digits=1,
                                                precision=1,
                                                sign=False,
                                                )
            return string.replace("+","")
        elif isinstance(obj, float):
            if abs_obj >= 10:
                return f"{obj:.0g}"
            else:
                return f"{obj:.2g}"
        else:
            return str(obj)
    elif isinstance(obj, type):
        name = obj.__name__
        return name[:min(len(name), 3)]
    elif isinstance(obj, list):
        return "[" + ",".join([str(val) for val in obj]) + "]"
    else:
        string = str(obj)
        return string[:min(len(string),3)]

# TODO turn selected Delta of configs into diff dict and then group name
# possbible entries in Delta:
# iterable_item_added iterable_item_moved iterable_item_removed
# set_item_added set_item_removed dictionary_item_added dictionary_item_removed
# attribute_added attribute_removed type_changes values_changed
# iterable_items_added_at_indexes iterable_items_removed_at_indexes
def to_short_string(dct: dict, ref_dct: dict=None):
    # assumes sorted dicts for now, to get same thing each time
    if ref_dct and isinstance(ref_dct, dict):
        keys = list(ref_dct.keys())
        for key in dct.keys():
            if key not in keys:
                keys.append(key)
    else:
        keys = list(dct.keys())
        ref_dct = {}

    key2str = find_short_strs(keys)
    ret_str = ""
    for key, val in dct.items():
        ret_str += key2str[key]
        ret_str += obj2shortstr(val, info=ref_dct.get(key))
        ret_str += "|"
    # remove last |
    ret_str = ret_str[:-1]
    return ret_str


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

def confidence(data, conf_level=0.95):
    data = np.asarray(data)
    # assume multiple data points, return as if, if 1d
    if len(data) <= 0:
        return None, None
    elif len(data) <= 1:
        return data.flatten(), np.zeros_like(data.flatten())
    m = np.mean(data, 0) # assumes to comptute over first dim
    s = np.std(data, 0, ddof=1)
    n = len(data)
    alpha = 1-conf_level
    t = stats.t.ppf(1-(alpha/2), df=n-1)

    return m, t * (s / np.sqrt(n)) # return mean and margin to add +-

def mess2flatnumpy(lst):
    try:
        return np.array(lst).flatten()
    except:
        pass
    ret = []
    for element in lst:
        ret.append(mess2flatnumpy(element))
    return np.concatenate(ret)

def data_relative_lims(data, ax, mi=None, ma=None):
    data = mess2flatnumpy(data)
    maxlim = np.max(data)
    minlim = np.min(data)
    rang = maxlim-minlim
    maxlim += np.abs(rang*0.3)
    minlim -= np.abs(rang*0.3)
    if mi is not None:
        minlim = np.min([mi, minlim])
    if ma is not None:
        maxlim = np.man([ma, maxlim])
    ax.set_ylim(minlim, maxlim)
    return minlim, maxlim

def put_some_last(names, put_last):
    add_last = []
    new_names = []
    for name in names:
        label = name.replace("eval_logs_","")
        if any(elem in label for elem in put_last):
            add_last.append(name)
        else:
            new_names.append(name)
    new_names += add_last
    names = new_names
    return names, add_last

def dict2string(dictionary, div="|"):
    options = ""
    pre = ""
    for key, val in dictionary.items():
        if isinstance(val, list):
            options += pre + key + "[" + ",".join(str(v) for v in val) + "]"
        elif isinstance(val, dict):
            options += pre + key + "{" + dict2string(val,div=",") + "}"
        elif isinstance(val, bool):
            if val is True:
                options += pre + key
            else:
                options += pre + key + "False"
        elif isinstance(val, (float, int)) and (val <= 1e-2 or val >= 9e3):
            options += pre + key + np.format_float_scientific(val,trim="-",exp_digits=1)
        else:
            options += pre + key + str(val)
        pre = div
    return options

def collect_dfr_data(experiments, eval_type="eval_logs", func=None):
    # this is only for base-rl
    values = []
    means = []
    disc_fut_rewards = []
    disc_fut_reward_means = []
    Vs = []
    V_means = []
    for exp in experiments:
        if not os.path.exists(os.path.join(exp, eval_type)):
            return None, None
        #avg_data = np.convolve(data, [1]*window, 'valid')/window
        # monitor is sequential, so we need to bundle it by number of experiments
        # in each evaluation, that info is found in completed.txt for now
        # TODO replace this with data from evaluations.npz when it collects correctly
        if os.path.isfile(os.path.join(exp, "completed.txt")):
            with open(os.path.join(exp, "completed.txt"),'r') as measure_info:
                reader = csv.reader(measure_info, delimiter=' ')
                m_info = {k: int(v) for [k, v] in reader}
            # TODO replace every use of monitor.csv IT IS WRONG!
            tot_reward = np.loadtxt(os.path.join(exp, eval_type, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
            time = np.loadtxt(os.path.join(exp, eval_type, "monitor.csv"), delimiter=',', skiprows=2, usecols=1)
            time = time.astype(int)
            print(len(time))
            print(time)
            gamma = 0.95
            #disc_fut_reward = tot_reward*(gamma**time)
            # For now, compute to reverse engineer -1 every step
            end_rew = tot_reward+time-1
            d_end_rew = end_rew*(gamma**time)
            d_step_cost = np.zeros_like(tot_reward)
            d_st_cur = -1
            d_st_tot = d_st_cur
            for i in range(1, int(np.max(time))):
                mask = time == i+1 # not for last step
                d_step_cost[mask] = d_st_tot
                d_st_cur = d_st_cur*0.99
                d_st_tot += d_st_cur
            disc_fut_reward = d_step_cost + d_end_rew

            if n_eval > 1:
                mean = means_for_multi_measure(tot_reward, m_info[eval_type], func=func)
                disc_fut_reward_mean = means_for_multi_measure(disc_fut_reward, m_info[eval_type], func=func)
            else:
                if func is not None:
                    mean = func(tot_reward)
                    disc_fut_reward_mean = func(disc_fut_reward)
                else:
                    mean = tot_reward
                    disc_fut_reward_mean = disc_fut_reward

            means.append(mean)
            values.append(tot_reward)
            disc_fut_rewards.append(disc_fut_reward)
            disc_fut_reward_means.append(disc_fut_reward_mean)

            if os.path.exists(os.path.join(exp, eval_type, "evaluations.npz")):
                evals = np.load(os.path.join(exp, eval_type, "evaluations.npz"))
                # for eval in evals:
                #     print(eval)
                #     print(evals[eval].shape)
                #     print(evals[eval])
                dataV = evals["initial_v"].flatten()
                dataV_mean = means_for_multi_measure(dataV, n_eval=n_eval, func=None)

            Vs.append(dataV)
            V_means.append(dataV_mean)

    return means, values, disc_fut_reward_means, disc_fut_rewards, V_means, Vs

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

def no_smooth(data, window):
    return data

smooth = gaussian_window_smooth
# smooth = window_smooth
# smooth = ema_smooth
#smooth = no_smooth

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
    if len(goals.shape) != 2 or len(coord_names) != len(goals[0,:]):
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
    # TODO handle when user accidentally give string, not list of strings
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
            # TODO replace every use of monitor.csv IT IS WRONG!
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
        # TODO replace every use of monitor.csv IT IS WRONG!
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
    mean, conf = confidence(means)
    mean = smooth(mean, window)
    conf = smooth(conf, window)
    if means.ndim > 1: # TODO remove and make separate tolerance method when for when I want it
        data = np.mean(means, axis=0)
        std = np.std(means, axis=0)
    else:
        data = means
        std = np.zeros_like(means)
    avg_data = smooth(data, window)
    avg_std = smooth(std, window)
    x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
    if name != None:
        ax.plot(x, mean, label=str(len(means))+ "exps " + f" {np.mean(data):.3f} mean " + name)
    else:
        ax.plot(x, mean, label=str(len(means))+ "exps " + f" {np.mean(data):.3f} mean " + os.path.basename(path))
    ax.fill_between(x, mean+conf, mean-conf, alpha=0.07,) # alpha=0.03
    return mean

def plot_individual_experiments(path,
                                window,
                                eval_type="eval_logs",
                                figname=None,
                                func=None,
                                name_append="",
                                ):
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
    mean, conf = confidence(means)
    mean = smooth(mean, window)
    conf = smooth(conf, window)
    if means.ndim > 1:
        data = np.mean(means, axis=0)
        std = np.std(means, axis=0)
    else:
        data = means
        std = np.zeros_like(means)
    avg_data = smooth(data, window)
    avg_std = smooth(std, window)
    x = np.linspace(0, len(avg_data)*eval_freq, len(avg_data))
    ax.plot(x, mean, label="average", zorder=100)
    # if name != None:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + name)
    # else:
    #     ax.plot(x, avg_data, label=str(len(experiments))+ "exps " + os.path.basename(path))
    ax.fill_between(x, mean+conf, mean-conf, alpha=0.07, zorder=100) # alpha=0.03
    minlim, maxlim = data_relative_lims([mean, means], ax)

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
    plt.savefig(os.path.join(path,figname+name_append), bbox_inches="tight")
    plt.close(fig)

def plot_each_goal_in_exp(path,
                          window,
                          n_eval = 1,
                          lst=["eval_logs", "train_logs"],
                          put_last = ['[-1.7, -0.02]', '[0.5, 0.02]'],
                          exclude=True,
                          func=None,
                          figname=None,
                          name_append=""
                          ):
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
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver', 'gold']) *
    #                     cycler('linestyle', ['-', ':', '--', '-.',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver',]) *
                    cycler('linestyle', [':',]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))
    # TODO find out why this isn't applying

    goal_prop = iter(cycler('color', ['k']) *
                    cycler('linestyle', ['--', '-.', (5, (10, 3))]))

    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=28)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

    datas = []
    disd_fut_rewards = []
    # if exclude is True:
    #     numbers = {s: int(s.replace("eval_logs_", "")) for s in names}
    #     names = sorted(names, key=numbers.__getitem__)
    names, add_last = put_some_last(names, put_last)

    for i, name in enumerate(names):
        if not os.path.exists(os.path.join(path, name)):
            return None, None
        # TODO replace every use of monitor.csv IT IS WRONG!
        data = np.loadtxt(os.path.join(path, name, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
        time = np.loadtxt(os.path.join(path, name, "monitor.csv"), delimiter=',', skiprows=2, usecols=1)
        gamma = 0.95
        disc_fut_reward = data*(gamma**time)
        if n_eval > 1:
            data = means_for_multi_measure(data, n_eval=n_eval, func=func)
            disc_fut_reward = means_for_multi_measure(disc_fut_reward, n_eval=n_eval, func=func)
        else:
            if func is not None:
                data = func(data)
        data_smooth = smooth(data, window)
        avg_disc_fut_reward = smooth(disc_fut_reward, window)
        x = np.linspace(0, len(data_smooth)*eval_freq, len(data_smooth))
        label = name.replace("eval_logs_","")
        label = label.replace("[","(")
        label = label.replace("]",")")
        if "(-1.7," in label: # hardcoded override
            label = " hard ext. goal"
        elif "(0.5," in label: # hardcoded override
            label = " easy ext. goal"
        # if "47" in label: # hardcoded override
        #     label += " (ext. goal)"
        # if "15" in label: # hardcoded override
        #     label += " (ext. goal)"
        else:
            label = str(i) # TODO temp override for large set of long goals
        if name in add_last:
            ax.plot(x, data_smooth, **next(goal_prop), label=label)
        else:
            ax.plot(x, data_smooth, label=label, alpha=0.7)
        datas.append(data)
        disd_fut_rewards.append(disc_fut_reward)

    # TODO determine if this should also be confidence, not tolerance
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

    meansD = np.mean(disd_fut_rewards, axis=0)
    stdD = np.std(disd_fut_rewards, axis=0)
    avg_dataD = smooth(meansD, window)
    if len(names) > 1:
        avg_stdD = smooth(stdD, window)
    else:
        avg_stdD = np.zeros_like(avg_dataD)

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, 0.0), # (0.65, 0.1), cliff #
              prop={'size': 11.5}, # 11.5 patho # 14 frozen # 11 cliff
              fancybox=True,
              ncol=4,
              framealpha=0.3).set_zorder(101)
    ax.set_ylabel("goal success rate")
    ax.set_xlabel("steps")
    #ax.set_ylim(-0.1, 1.05) # ax.set_ylim(-0.05, 1.05) cliff # ax.set_ylim(-0.2, 1.05) pmc
    ax.set_ylim(-0.05, 1.05)

    plt.savefig(os.path.join(path,figname+name_append), bbox_inches="tight")
    plt.close(fig)

    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver',]) *
                    cycler('linestyle', [':',(0, (3, 1))]))) # ':', '--', '-.', (5, (10, 3)) '--', '-.', (5, (10, 3))

    goal_prop = iter(cycler('color', ['k']) *
                cycler('linestyle', ['--', '-.', (5, (10, 3))]))

    datas = []
    # TODO temp prep fig for V data too
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    for i, name in enumerate(names):
        if not os.path.exists(os.path.join(path, name)):
            return avg_data, avg_std
        # try to get and show initial_v data TODO
        if not os.path.exists(os.path.join(path, name, "evaluations.npz")):
            return avg_data, avg_std
        evals = np.load(os.path.join(path, name, "evaluations.npz"))
        lst = evals.files
        data = evals["initial_v"].flatten()
        label = name.replace("eval_logs_","")
        label = label.replace("[","(")
        label = label.replace("]",")")
        if "(-1.7," in label: # hardcoded override
            label = "hard ext. goal"
        elif "(0.5," in label: # hardcoded override
            label = "easy ext. goal"
        # if "47" in label: # hardcoded override
        #     label += " (ext. goal)"
        # if "15" in label: # hardcoded override
        #     label += " (ext. goal)"
        else:
            label = str(i) # TODO temp override for large set of long goals
        if n_eval > 1:
            data = means_for_multi_measure(data, n_eval=n_eval, func=None)
        avg_dataV = smooth(data, window)
        x = np.linspace(0, len(avg_dataV)*eval_freq, len(avg_dataV))
        disd_fut_reward_smooth = data_smooth = smooth(disd_fut_rewards[i], window)
        if name in add_last:
            prop = next(goal_prop)
            ax.plot(x, avg_dataV, **prop, label=label+" V est.")
            prop['color'] = 'red'
            ax.plot(x, disd_fut_reward_smooth, **prop, label=label+" real")
        else:
            ax.plot(x, avg_dataV, label=label+" V est.", alpha=0.5)
            ax.plot(x, disd_fut_reward_smooth, label=label+" real", alpha=0.5)
        datas.append(data)

    meansV = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    avg_dataV = smooth(meansV, window)
    if len(names) > 1:
        avg_stdV = smooth(std, window)
        x = np.linspace(0, len(avg_dataV)*eval_freq, len(avg_dataV))
        ax.plot(x, avg_dataV, label="average V est.", zorder=100, color='red', linestyle='-')
        ax.fill_between(x, avg_dataV+avg_stdV, avg_dataV-avg_stdV, alpha=0.05, zorder=1, color='red')
        ax.plot(x, avg_dataD, label="average real", zorder=100, color='k', linestyle='-')
        ax.fill_between(x, avg_dataD+avg_stdD, avg_dataD-avg_stdD, alpha=0.05, zorder=1, color='k')
    else:
        avg_stdV = np.zeros_like(avg_dataV)

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, 0.0), # (0.65, 0.1), cliff #
              prop={'size': 11.5}, # 11.5 patho # 14 frozen # 11 cliff
              fancybox=True,
              ncol=4,
              framealpha=0.3).set_zorder(101)
    ax.set_ylabel("(expected) future discounted reward")
    ax.set_xlabel("steps")
    #minlim, maxlim = data_relative_lims([avg_dataD, avg_dataV, datas, disd_fut_rewards], ax)
    #ax.set_ylim(-0.2, 1.05) # ax.set_ylim(-0.05, 1.05) cliff
    ax.set_ylim(-0.05, 1.05)

    plt.savefig(os.path.join(path,"initial_V_vs_discounted_future_reward"+name_append), bbox_inches="tight")
    plt.close(fig)

    # TODO return and make meta plot for V data too, but maybe make own func instead then...
    return means, meansV, meansD

def plot_all_in_folder(dir,
                       coord_names,
                       num2keep=-1,
                       keywords=[],
                       filter=None,
                       name_override=None,
                       name_append="",
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
    # folders_end = []
    # for i, str in enumerate(folders):
    #     if ("base-rl" in str):
    #         folders_end.append(folders[i])
    # folders = [x for x in folders if x not in folders_end]
    # folders += folders_end
    folders, add_last = put_some_last(folders, ["base-rl", "epsilon"])

    avg_datas = []
    for i, folder in enumerate(folders):
        if name_override != None:
            avg_data = add_subplot(folder, window, ax, eval_type=eval_type, name=name_override[i], func=func)
        else:
            avg_data = add_subplot(folder, window, ax, eval_type=eval_type, func=func)
        avg_datas.append(avg_data)

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
    minlim, maxlim = data_relative_lims(avg_datas, ax)
    fig.tight_layout()
    name = ""
    if num2keep > 0:
        name += f"top{num2keep}_"
    name += eval_type
    if keywords != []:
        name += f"_including_{keywords}"
    if func is not None:
        name += "_" + "func"
    plt.savefig(os.path.join(dir, name+name_append))
    plt.close(fig)

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

def plot_base_rl_dfr(names,
                     num_exps,
                     meansV,
                     confsV,
                     meansD,
                     confsD,
                     window,):
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

    for name, num_exp, meanV, confV, meanD, confD in zip(names, num_exps, meansV, confsV, meansD, confsD):
        if meanV.size == 0:
            break
        x = np.linspace(0, len(meanV)*eval_freq, len(meanV))
        print(meanD)
        ax.plot(x, meanV, label=str(num_exp)+ "exps " + f" {np.mean(meanV):.3f} mean " + name + " V ext.", zorder=2)
        if False: # tolerance
            ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)
        else: # confidence
            ax.fill_between(x, meanV+confV, meanV-confV, alpha=0.07, zorder=1)
        ax.plot(x, meanD, label=str(num_exp)+ "exps " + f" {np.mean(meanD):.3f} mean " + name + " real", zorder=2)
        if False: # tolerance
            ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)
        else: # confidence
            ax.fill_between(x, meanD+confD, meanD-confD, alpha=0.07, zorder=1)


    ax.legend(loc='upper right', # 'upper left' # 'lower right'
              prop={'size': 7}, # 8 # 18
              fancybox=True,
              framealpha=0.3).set_zorder(101)#, bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("(expected) future discounted reward")
    ax.set_xlabel("steps")
    #minlim, maxlim = data_relative_lims(mean, ax, mi=0)
    #ax.set_ylim([-0.2, 1.1])
    #ax.set_ylim([-1.2, 2.1]) #test showing more when few data
    #ax.set_ylim(np.max([-0.05,minlim]), np.min([1.05,maxlim]))

    plt.savefig(os.path.join(folder,"base-rl_initial_V_vs_discounted_future_reward"+name_append), bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--top', default=0)
    parser.add_argument('-k', '--keep', nargs='*', default=[])
    parser.add_argument('-f', '--filter', nargs='*', default=[])
    parser.add_argument('-n', '--name_override', nargs='*', default=None)
    parser.add_argument('-g', '--goal_plots', action='store_true')
    parser.add_argument('-c', '--cutoff', type=int, default=None)
    parser.add_argument('-s', '--smooth', default=None,
                        help="If not set, no smoothing. Smoothing options: \
                            gaussian/g, window/w, ema/e")
    parser.add_argument('-r', '--random_goals_only', action='store_true')
    args = parser.parse_args()

    #folder, coord_names = "./output/wrapper/SparsePendulumEnv-v1", ["x", "y", "ang. vel."],
    folder, coord_names = "./output/wrapper/PathologicalMountainCar-v1.1", ["xpos", "velocity"], # _high_freq_eval
    #folder, coord_names = "./output/wrapper/FrozenLake-v1", ["index"], # TODO replace with grid reshape
    #folder, coord_names = "./output/wrapper/CliffWalking-v0", ["index"]

    #folder, coord_names = "./output/wrapper/PathologicalMountainCar-v1.1_outdated_goal_selection_versions/intermediate_prior_to_0.5_factor_in_target_vs_visits", ["xpos", "velocity"]

    #eval_freq = 100000
    #eval_freq = 50000 #50000 frozen, 20000 for patho MC, (high freq 10000)
    if "FrozenLake" in folder:
        eval_freq = 10_000
        thresh = 1
        symlog_y = False
        n_eval = 5
        put_last = ['15']
    elif "Patho" in folder:
        eval_freq = 200_000# 50000
        thresh = 11
        symlog_y = False
        n_eval = 1
        put_last = ['[-1.7, -0.02]', '[0.6, 0.02]',
                    '[-1.7, -0.07, -1.6, 0.0]', '[0.5, 0.0, 0.6, 0.07]','epsilon']
    elif "Cliff" in folder:
        eval_freq = 50_000
        thresh = -14.0
        symlog_y = True
        n_eval = 1
        put_last = ['47']
    eval_f_adjust = 50_000/eval_freq

    if args.smooth in ["gaussian", "g"]:
        smooth = gaussian_window_smooth
        name_append = "_gaussian_smoothed"
    elif args.smooth in ["window", "w"]:
        smooth = window_smooth
        name_append = "_window_smoothed"
    elif args.smooth in ["ema", "e"]:
        smooth = ema_smooth
        name_append = "_ema_smoothed"
    else:
        smooth = no_smooth
        name_append = ""

    setups = get_all_folders(folder, full=True)
    names, paths = zip(*setups)
    names = get_names_with(names, ["base-rl-baseline"])
    paths = get_names_with(paths, ["base-rl-baseline"])
    meanVs = []
    confVs = []
    meanDs = []
    confDs = []
    num_exps = []
    for name, path in zip(names, paths):
        print(f"name {name}")
        print(f"path {path}")
        exps = get_all_folders(path)
        exps, add_last = put_some_last(exps, put_last)
        # for exp in exps:
        #     if args.random_goals_only:
        #         lst = ["eval_logs", "train_logs",]
        #         for entry in put_last:
        #             lst.append("eval_logs_"+entry)
        #     else:
        #         lst = ["eval_logs", "train_logs"]
        dfr_data = collect_dfr_data(exps, "eval_logs")
        means, values, disc_fut_reward_means, disc_fut_rewards, V_means, Vs = dfr_data
        meanV, confV = confidence(V_means)
        # meanV = smooth(meanV, int(20*eval_f_adjust))
        # confV = smooth(confV, int(20*eval_f_adjust))
        meanD, confD = confidence(disc_fut_reward_means)
        meanVs.append(meanV)
        confVs.append(confV)
        meanDs.append(meanD)
        confDs.append(confD)
        num_exps.append(len(exps))

        # TODO add supplot for each here

    #meanV, confV = confidence(meanVs)
    #meanD, confD = confidence(meanDs)
    plot_base_rl_dfr(names,
                     num_exps,
                     meanVs,
                     confVs,
                     meanDs,
                     confDs,
                     window=int(20*eval_f_adjust), # 20
                    )

    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logs",
                       cutoff=args.cutoff,
                       window=int(20*eval_f_adjust), #10
                       symlog_y=symlog_y,
                       goal_plots=args.goal_plots,
                       name_append=name_append,
                       )
    plot_all_in_folder(folder,
                       coord_names = coord_names,
                       num2keep=int(args.top),
                       keywords=args.keep,
                       filter=args.filter,
                       name_override=args.name_override,
                       eval_type="eval_logs",
                       cutoff=args.cutoff,
                       window=int(20*eval_f_adjust), #10
                       indi_plots=False,
                       func=threshold_gen(thresh),
                       name_append=name_append,
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
    means = []
    confs = []
    meansV = []
    confsV = []
    meansD = []
    confsD = []
    for name, path in setups:
        exps = get_all_folders(path)
        exp_datas = []
        exp_edfr = []
        exp_dfr = []
        put_last = ["epsilon"]
        exps, add_last = put_some_last(exps, put_last)
        for exp in exps:
            if args.random_goals_only:
                lst = ["eval_logs", "train_logs",]
                for entry in put_last:
                    lst.append("eval_logs_"+entry)
            else:
                lst = ["eval_logs", "train_logs"]
            # plot goal successes
            avg_data, avg_edfr, avg_dfr = plot_each_goal_in_exp(exp,
                                                      window=int(20*eval_f_adjust), # 20
                                                      n_eval=n_eval,
                                                      put_last=put_last,
                                                      lst=lst,
                                                      figname="goal_success_rate",
                                                      name_append=name_append,
                                                      )
            # plot training success rate
            plot_each_goal_in_exp(exp,
                                  window=100,
                                  n_eval=1,
                                  lst=["train_logs"],
                                  exclude=False,
                                  func=threshold_gen(1), # TODO replace with non pmc hardcoded
                                  figname="train_success_rate",
                                  name_append=name_append,
                                  )
            if avg_data is None or avg_data.shape == ():
                continue
            exp_datas.append(avg_data)
            exp_edfr.append(avg_edfr)
            exp_dfr.append(avg_dfr)
        # TODO this is operating on smoothed data (when on) and that seems wrong?
        # any smoothing should probably be after call to confidence
        mean, conf = confidence(exp_datas)
        mean = smooth(mean, int(20*eval_f_adjust))
        conf = smooth(conf, int(20*eval_f_adjust))
        meanV, confV = confidence(exp_edfr)
        meanV = smooth(meanV, int(20*eval_f_adjust))
        confV = smooth(confV, int(20*eval_f_adjust))
        meanD, confD = confidence(exp_dfr)
        meanD = smooth(meanD, int(20*eval_f_adjust))
        confD = smooth(confD, int(20*eval_f_adjust))
        if len(exps) > 1:
            exp_datas = np.mean(exp_datas, 0)
            exp_edfr = np.mean(exp_edfr, 0)
        else:
            exp_datas = np.asarray(exp_datas).flatten()
            exp_edfr = np.asarray(exp_edfr).flatten()
        if exp_datas.shape != (): # handle empty experiment list (unfinished exp)
            avg_datas.append(exp_datas)
            avg_stds.append(exp_edfr)
            names.append(name)
            num_exps.append(len(exps))
            means.append(mean)
            confs.append(conf)
            meansV.append(meanV)
            confsV.append(confV)
            meansD.append(meanD)
            confsD.append(confD)

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k']) +
    #                         cycler('linestyle', ['-', ':','--', '-.'])))
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver']) *
                        cycler('linestyle', ['-',':',]))) # ':', '--', '-.', (5, (10, 3))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

    for name, data, num_exp, mean, conf in zip(names, avg_datas, num_exps, means, confs):
        if data.size == 0:
            break
        x = np.linspace(0, len(data)*eval_freq, len(data))
        ax.plot(x, mean, label=str(num_exp)+ "exps " + f" {np.mean(mean):.3f} mean " + name, zorder=2)
        if False: # tolerance
            ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)
        else: # confidence
            ax.fill_between(x, mean+conf, mean-conf, alpha=0.07, zorder=1)

    # names = ["Intermediate Success", "Novelty",
    #          "Uniform Random",]
    # for i, name in enumerate(names):
    #     ax.lines[i].set_label(name)

    ax.legend(loc='lower right', # 'upper left' # 'lower right'
              prop={'size': 7}, # 8 # 18
              fancybox=True,
              framealpha=0.3).set_zorder(101)#, bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("goal success rate")
    ax.set_xlabel("steps")
    minlim, maxlim = data_relative_lims(mean, ax, mi=0)
    #ax.set_ylim([-0.2, 1.1])
    #ax.set_ylim([-1.2, 2.1]) #test showing more when few data
    ax.set_ylim(np.max([-0.05,minlim]), np.min([1.05,maxlim]))

    plt.savefig(os.path.join(folder,"average_goal_success_rate"+name_append), bbox_inches="tight")
    plt.close(fig)


    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'sienna', 'pink', 'palegreen', 'silver']) *
                        cycler('linestyle', ['-',':','--','-.',]))) # ':','--','-.',(5, (10, 3))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

    for name, num_exp, meanV, confV, meanD, confD in zip(names, num_exps, meansV, confsV, meansD, confsD):
        if meanV.size == 0:
            break
        x = np.linspace(0, len(meanV)*eval_freq, len(meanV))
        ax.plot(x, meanV, label=str(num_exp)+ "exps " + f" {np.mean(meanV):.3f} mean " + name + " V ext.", zorder=2)
        if False: # tolerance
            ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)
        else: # confidence
            ax.fill_between(x, meanV+confV, meanV-confV, alpha=0.07, zorder=1)
        ax.plot(x, meanD, label=str(num_exp)+ "exps " + f" {np.mean(meanD):.3f} mean " + name + " real", zorder=2)
        if False: # tolerance
            ax.fill_between(x, data+std, data-std, alpha=0.07, zorder=1)
        else: # confidence
            ax.fill_between(x, meanD+confD, meanD-confD, alpha=0.07, zorder=1)

    # names = ["Intermediate Success", "Novelty",
    #          "Uniform Random",]
    # for i, name in enumerate(names):
    #     ax.lines[i].set_label(name)

    ax.legend(loc='upper right', # 'upper left' # 'lower right'
              prop={'size': 7}, # 8 # 18
              fancybox=True,
              framealpha=0.3).set_zorder(101)#, bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("(expected) future discounted reward")
    ax.set_xlabel("steps")
    minlim, maxlim = data_relative_lims(mean, ax, mi=0)
    #ax.set_ylim([-0.2, 1.1])
    #ax.set_ylim([-1.2, 2.1]) #test showing more when few data
    ax.set_ylim(np.max([-0.05,minlim]), np.min([1.05,maxlim]))

    plt.savefig(os.path.join(folder,"initial_V_vs_discounted_future_reward"+name_append), bbox_inches="tight")
    plt.close(fig)