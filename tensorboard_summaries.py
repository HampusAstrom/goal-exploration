import os
from collections import defaultdict
import argparse
import re

import numpy as np
import scipy.stats as stats
import torch.utils.tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# inspired by same in utils, but for single val, maybe move there TODO?
def confidence(data, conf_level=0.95, axis=-1):
    data = np.asarray(data)
    # assume multiple data points, return as if, if 1d
    if len(data) <= 0:
        return None, None
    elif len(data) <= 1:
        return data.flatten(), np.zeros_like(data.flatten())
    m = np.mean(data, axis=axis) # assumes to comptute over last dim
    s = np.std(data, axis=axis, ddof=1)
    n = len(data)
    alpha = 1-conf_level
    t = stats.t.ppf(1-(alpha/2), df=n-1)

    return m, t * (s / np.sqrt(n)) # return mean and margin to add +-

def first_chars_after(string, num=1, delim="/"):
    return string.split(delim,1)[1][0:num]

def purge_in_path(path, start="events.out"):
    return
    for file in os.listdir(path):
        if file.startswith("events.out"):
                    path_file = os.path.join(path, file)
                    print(path_file)
                    os.remove(path_file)

def regex_each(regex, s_lst, keep=True, op=re.search):
    match_lst = []
    for s in s_lst:
        if keep and op(regex, s):
            match_lst.append(s)
        elif not keep and op(regex, s):
            match_lst.append(s)
    if keep:
        return match_lst
    else:
        return [s for s in s_lst if s not in match_lst]

def filter_dicts_by_keys(keys, dict_list):
    # TODO should this be dict of dicts as input and output instead?
    new_dict_list = []
    for dict in dict_list:
        new_dict = {key: dict[key] for key in keys}
        new_dict_list.append(new_dict)
    return new_dict_list

def dict2numpy(dict):
    print(len(dict))
    tags, vals = zip(*dict.items())
    return tags, np.array(vals)

def get_mean_of_means(dicts, single_measure=False):
    # TODO handle if only one exp

    # for average compare: mean over all goals first, then calc mean/conf over exps
    tags, vals = dict2numpy(dicts[0])
    _, steps = dict2numpy(dicts[1])
    _, wc_times = dict2numpy(dicts[2])

    print(vals.shape)
    if not single_measure:
        means_1, _ = confidence(vals, axis=0) # mean over tags
        mean_steps, _ = confidence(steps, axis=0)
        mean_wc_times, _ = confidence(wc_times, axis=0)
        print(means_1.shape)
    else:
        means_1 = vals[0]
        print(f"{means_1.shape} means_1.shape")
        mean_steps = steps[0]
        mean_wc_times = wc_times[0]
    means, confs = confidence(means_1, axis=-1) # mean over exps
    mean_steps, _ = confidence(mean_steps, axis=-1)
    mean_wc_times, _ = confidence(mean_wc_times, axis=-1)
    print(means.shape)
    print()
    print(mean_steps.shape)
    print(mean_wc_times.shape)
    print()
    return means, confs, mean_steps, mean_wc_times

def write_confidences(tag, means, confs, steps, wc_times, writer):
    upper_tag = "hide_sum_eval/"+tag+"_upper_lim"
    mean_tag  = "hide_sum_eval/"+tag+"_mean"
    lower_tag = "hide_sum_eval/"+tag+"_lower_lim"
    print("in write_confidences")
    print(upper_tag + "   " + mean_tag + "   " + lower_tag)
    print(means.shape)
    print(confs.shape)
    print(steps.shape)
    print(wc_times.shape)
    for mean, conf, step, wc_time in zip(means, confs, steps, wc_times):
        writer.add_scalar(upper_tag, mean+conf,np.mean(step),np.mean(wc_time))
        writer.add_scalar(mean_tag, mean,np.mean(step),np.mean(wc_time))
        writer.add_scalar(lower_tag, mean-conf,np.mean(step),np.mean(wc_time))
    return {"sum_with_conf_"+tag: ["Margin", [mean_tag, upper_tag, lower_tag]]}

def tabulate_events(dpath):
    print(os.listdir(dpath))
    lst = []
    for dname in os.listdir(dpath):
        # only add from exp folders that are completed
        if "exp" in dname and os.path.isfile(os.path.join(dpath, dname, "completed.txt")):
            lst.append(dname)
    print(lst)
    event_accumulators = []
    for dname in lst:
        sum_path = os.path.join(dpath, dname, "train_logs")
        # For now we want the latest directory, later we migh just want all instead
        subdirs = [f.path for f in os.scandir(sum_path) if f.is_dir()]
        newest_file = max(subdirs, key=os.path.getctime)
        event_accumulators.append(EventAccumulator(newest_file).Reload())

    tags = event_accumulators[0].Tags()['scalars']
    # print(tags)
    # print(tags[0])
    #print(summary_iterators[0].Scalars(tags[0]))

    for it in event_accumulators:
        assert it.Tags()['scalars'] == tags

    # check length of each tag in each exp
    uneven_len_tags = []
    for tag in tags:
        lens = []
        for acc in event_accumulators:
            lens.append(len(acc.Scalars(tag)))
        #print(f"{tag} {lens} {(lens[0] == np.array(lens)).all()}")
        if not (lens[0] == np.array(lens)).all():
            uneven_len_tags.append(tag)
    tags = list(set(tags) - set(uneven_len_tags)) # for now, skip all uneven

    # TODO make work for non-eval stuff
    # for now we only use tags with eval
    non_eval_tags = []
    for tag in tags:
        if "eval" not in tag:
            non_eval_tags.append(tag)
    tags = list(set(tags) - set(non_eval_tags))

     # TODO handle uneven lengths, maybe by trying to grab the closest in
     # time each time or something? tricky, ignore for now

    values = defaultdict(list) # TODO replace with new log tracker?
    steps = defaultdict(list)
    wall_times = defaultdict(list)

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in event_accumulators]):
            # check that all stepsize are the same, TODO remove when we can handle uneven
            if len(set(e.step for e in events)) != 1:
                print(dpath)
                print(f"{tag} {[e.step for e in events]}")
            assert len(set(e.step for e in events)) == 1

            values[tag].append([e.value for e in events])
            steps[tag].append([e.step for e in events])
            wall_times[tag].append([e.wall_time for e in events])

    # There could be a need to sort the outer lists here based on steps,
    # to make sure they are in the right order
    return values, steps, wall_times


def write_exp_mean_conf_events(dpath, vals, steps, wc_times, dname='combined'):

    fpath = os.path.join(dpath, dname)
    # TODO remove old events.out.tfevents... here before creating new
    # TODO replace "eval/" and similar tag with "eval_sum/" and similar
    # TODO and maybe also replace upper and lower tags with different "eval_X/",
    # to hide them away

    tags, values = zip(*vals.items())
    t_steps, steps = zip(*steps.items())
    t_wc, wc_times = zip(*wc_times.items())

    assert tags == t_steps and tags == t_wc # make sure all are ordered the same

    V_means, V_confs = confidence(values)
    T_means, _ = confidence(steps)
    WC_means, _ = confidence(wc_times)

    print(np.array(values).shape)
    print(V_means.shape)
    purge_in_path(fpath)
    writer = tb.writer.SummaryWriter(fpath)
    margin_charts = {}
    for tag, Vmeans, Vconfs, Tmeans, WCmeans in zip(tags, V_means, V_confs, T_means, WC_means):
        upper_tag = "hide_"+tag+"_upper_lim"
        mean_tag = "hide_"+tag+"_mean"
        lower_tag = "hide_"+tag+"_lower_lim"
        for Vmean, Vconf, Tmean, WCmean in zip(Vmeans, Vconfs, Tmeans, WCmeans):
            # summary = tb.writer.SummaryWriter(value=[tb.writer.SummaryWriter.Value(tag=tag, simple_value=mean)])
            # writer.add_summary(summary, global_step=i)
            # writer.add_scalars(tag+dname,
            #                    {upper_tag: Vmean+Vconf,
            #                     mean_tag: Vmean,
            #                     lower_tag: Vmean-Vconf},
            #                    np.mean(Tmean),
            #                    np.mean(WCmean))
            # writer.add_scalars(tag+dname,
            #                    {upper_tag: Vmean+Vconf,
            #                     mean_tag: Vmean,
            #                     lower_tag: Vmean-Vconf},
            #                    np.mean(Tmean),
            #                    np.mean(WCmean))
            writer.add_scalar(upper_tag, Vmean+Vconf,np.mean(Tmean),np.mean(WCmean))
            writer.add_scalar(mean_tag, Vmean,np.mean(Tmean),np.mean(WCmean))
            writer.add_scalar(lower_tag, Vmean-Vconf,np.mean(Tmean),np.mean(WCmean))

        margin_charts["with_conf_"+tag] = ["Margin", [mean_tag, upper_tag, lower_tag]]
        # writer.add_custom_scalars_marginchart([upper_tag, mean_tag, lower_tag],
        #                                       category="test_category",
        #                                       title=tag)
    layout = {"all_with_confidences": margin_charts}
    writer.add_custom_scalars(layout)
    writer.flush()
    return margin_charts, tags, V_means, T_means, WC_means

def setup_dfr_vs_edfr_events(dfr_dicts, edfr_dicts, writer):
    # TODO do both singular comparisons and average compare
    dfr_means, dfr_confs, dfr_mean_steps, dfr_mean_wc_times = get_mean_of_means(dfr_dicts)
    edfr_means, edfr_confs, edfr_mean_steps, edfr_mean_wc_times = get_mean_of_means(edfr_dicts)
    initV_tag = "average_initial_V"
    initV_chart = write_confidences(initV_tag,
                                    edfr_means,
                                    edfr_confs,
                                    edfr_mean_steps,
                                    edfr_mean_wc_times,
                                    writer)
    dfr_tag = "average_discounted_reward"
    dfr_chart = write_confidences(dfr_tag,
                                  dfr_means,
                                  dfr_confs,
                                  dfr_mean_steps,
                                  dfr_mean_wc_times,
                                  writer)
    dfr_vs_edfr_chart = {"dfr_vs_edfr_chart": ["Multiline",["hide_sum_eval/"+initV_tag+"_mean",
                                                            "hide_sum_eval/"+dfr_tag+"_mean"]]}
    layout = {"expected_and_actual_discounted_reward": dfr_vs_edfr_chart | initV_chart | dfr_chart}
    return layout

def setup_base_rl_dfr_vs_edfr_events(dfr_dicts, edfr_dicts, writer):
    # TODO do both singular comparisons and average compare
    dfr_means, dfr_confs, dfr_mean_steps, dfr_mean_wc_times = get_mean_of_means(dfr_dicts, single_measure=True)
    edfr_means, edfr_confs, edfr_mean_steps, edfr_mean_wc_times = get_mean_of_means(edfr_dicts, single_measure=True)
    initV_tag = "base_rl_average_initial_V"
    initV_chart = write_confidences(initV_tag,
                                    edfr_means,
                                    edfr_confs,
                                    edfr_mean_steps,
                                    edfr_mean_wc_times,
                                    writer)
    dfr_tag = "base_rl_average_discounted_reward"
    dfr_chart = write_confidences(dfr_tag,
                                  dfr_means,
                                  dfr_confs,
                                  dfr_mean_steps,
                                  dfr_mean_wc_times,
                                  writer)
    dfr_vs_edfr_chart = {"base_rl_dfr_vs_edfr_chart": ["Multiline",["hide_sum_eval/"+initV_tag+"_mean",
                                                            "hide_sum_eval/"+dfr_tag+"_mean"]]}
    layout = {"expected_and_actual_discounted_reward": dfr_vs_edfr_chart | initV_chart | dfr_chart}
    return layout

def write_mean_goal_events(dpath, vals, steps, wc_times, dname='combined_special'):
    fpath = os.path.join(dpath, dname)
    purge_in_path(fpath)
    writer = tb.writer.SummaryWriter(fpath)
    # TODO remove old events.out.tfevents... here before creating new

    all_charts = {}

    # lets try regex instead:
    tags = [s for s in vals.keys()]
    tags = sorted(tags)
    new_tags = regex_each(r'(/\d)', tags) # only numbered goals
    # TODO make filter below an on/off thing via param to this method
    new_tags = regex_each("(/0_|/1_)", new_tags, False) # remove all non-random aka /0_ and /1_
    inv_tags = sorted(list(set(tags) - set(new_tags)))
    # print(inv_tags)
    # if "base-rl" in dpath:
    #     print(tags)
    #     exit()
    # print(*new_tags, sep='\n')
    # print()
    dfr_tags = regex_each("(episode_disc_rewards)", new_tags)
    dfr_dicts = filter_dicts_by_keys(dfr_tags,[vals, steps, wc_times])
    edfr_tags = regex_each("(initial_values)", new_tags)
    edfr_dicts = filter_dicts_by_keys(edfr_tags,[vals, steps, wc_times])
    if len(new_tags) != 0: # "base-eval" has no length, TODO just do discounted vs real on hard tags instead
        edfr_vs_dfr_charts = setup_dfr_vs_edfr_events(dfr_dicts,edfr_dicts, writer)
        all_charts = all_charts | edfr_vs_dfr_charts
    else:
        dfr_tags = regex_each("(episode_disc_rewards)", inv_tags)
        dfr_dicts = filter_dicts_by_keys(dfr_tags,[vals, steps, wc_times])
        edfr_tags = regex_each("(initial_values)", inv_tags)
        edfr_dicts = filter_dicts_by_keys(edfr_tags,[vals, steps, wc_times])
        if len(dfr_tags) > 1 or len(edfr_tags) > 0:
            print(*edfr_tags, sep='\n')
            print()
            print(*dfr_tags, sep='\n')
            print()
            edfr_vs_dfr_charts = setup_base_rl_dfr_vs_edfr_events(dfr_dicts,edfr_dicts, writer,)
            all_charts = all_charts | edfr_vs_dfr_charts

    mean_success_tags = regex_each("(mean_reward)", new_tags)
    # print(*edfr_tags, sep='\n')
    # print()
    # print(*dfr_tags, sep='\n')
    # print()
    # print(*mean_success_tags, sep='\n')
    # print()

    # print(f"{len(vals)} {len(steps)} {len(wc_times)}")

    # Gather initial V data

    # Gather discounted reward data

    # prep shared plot for initical V and discounted reward

    # Gather mean epiode length

    # Should I also plot mean episode length?
    print(all_charts)
    if len(all_charts) > 0:
        writer.add_custom_scalars(all_charts)
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1")
    args = parser.parse_args()
    path = args.name

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]

    # TODO handle that we don't overwrite the event logs we write here

    for dname in dirs:
        print(dname)
        dpath = os.path.join(path, dname)
        vals, steps, wc_times = tabulate_events(dpath)
        if len(vals) < 1:
            continue
        # TODO write_mean_goal_events should maybe use data returned above instead? no
        # or different for different?
        # per exp plots needed:     should I do per exp goals in separate or write_combined_events
        #   average of all goal reward (with conf or std?), with goals
        #   compare per goal initial V vs discounted future reward
        #   (per exp plots should maybe not be in tensorboard at all?)
        # per parent folder: <- focus on these for now (should be mean over exps of (mean over goals))
        #   average of all goal reward (this could be logged intead?)
        #   compare average of goal initial V vs (average of all goal) discounted future reward
        #   for base-rl: initial V vs discounted future reward instead of above
        write_mean_goal_events(dpath,vals,steps,wc_times)
        base_marg_charts, tags, V_mean, T_mean, WC_mean = write_exp_mean_conf_events(dpath,
                                                                          vals,
                                                                          steps,
                                                                          wc_times)
