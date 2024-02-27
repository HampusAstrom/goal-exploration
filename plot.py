import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# current selection
# paths = ["./output/pendulum/trunc_pendulum_eval/fixed_dense_genericnormed_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/fixed_dense_pendulum_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/fixed_truncated_genericnormed_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/fixed_truncated_pendulum_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/random_dense_genericnormed_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/random_dense_pendulum_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/random_truncated_genericnormed_10000/exp1/eval_logs",
#         "./output/pendulum/trunc_pendulum_eval/random_truncated_pendulum_10000/exp1/eval_logs",
#         ]
# labels = ["fixed_dense_genericnormed_10000",
#           "fixed_dense_pendulum_10000",
#           "fixed_truncated_genericnormed_10000",
#           "fixed_truncated_pendulum_10000",
#           "random_dense_genericnormed_10000",
#           "random_dense_pendulum_10000",
#           "random_truncated_genericnormed_10000",
#           "random_truncated_pendulum_10000",]

# linestyles = ["r-",
#              "r--",
#              "r:",
#              "r-.",
#              "b-",
#              "b--",
#              "b:",
#              "b-.",
#              ]

matplotlib.rcParams.update({'font.size': 18})

from itertools import product

goal = ["fixed", "random",]
density = ["truncated"] # ["truncated", "dense"]
reward_type = ["pendulum"] # ["pendulum","genericnormed"]
harder_start = [0.1, 0.2, 0.5, 1] # [0.1, 0.2, 0.5, 1]
experiments = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8",]

lineparts = product(["-", ":", "-", "-."], ["r", "g", "c", "b"]) # ["--", ":", "-", "-."], ["r", "y", "g", "b"]
colors = ["r", "g", "c", "b"]*4
linestyles = [x[0]+x[1] for x in lineparts]
skips = [0, 0, 0, 0, 0, 0, 0, 0]
#skips = [x == 1 for x in skips]

steps = 20000

print(colors)
print(linestyles)

def add_subplot(paths, labels, linestyles, colors, skips, endx, window, ax):
    for path, label, linestyle, color, skip in zip(paths, labels, linestyles, colors, skips):
        if skip:
            continue
        datas = []
        for exp in path:
            data = np.loadtxt(os.path.join(exp, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
            #avg_data = np.convolve(data, [1]*window, 'valid')/window
            datas.append(data)
        #avg_data = np.mean(datas, axis=0)
        #avg_std = np.std(datas, axis=0)
        data = np.mean(datas, axis=0)
        std = np.std(datas, axis=0)
        avg_data = np.convolve(data, [1]*window, 'valid')/window
        avg_std = np.convolve(std, [1]*window, 'valid')/window
        #x = range(len(avg_data))
        x = np.linspace(0, endx, len(avg_data))
        ax.plot(x, avg_data, linestyle, label=label)
        ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.03, facecolor=color)

paths = []
labels = []
base_path = "./output/pendulum/hardstart/truncated_pendulum_eval/"
for conf in product(goal, density, reward_type, harder_start):
    (goals, densitys, reward_types, harder_starts)=conf
    # experiment = "exp1"
    exp_set = []
    extra = ""
    # if goals == "fixed": # override for no hindsight 
    #     extra = "_noHerReplay"
    options = goals + "_" + densitys + "_" + reward_types + "_" + str(steps) + "steps_" + str(harder_starts) + "hardstart" + extra
    if goals == "fixed":
        goals = "fixed goal selection + hindsight"
    else:
        goals = "random goal selection + hindsight"
    short_options = str(float(harder_starts)) + " start, " + goals #+ " goal"
    for experiment in experiments:
        eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
        exp_set.append(eval_log_dir)
    paths.append(exp_set)
    #labels.append(options)
    labels.append(short_options)


px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
window = 10
add_subplot(paths, labels, linestyles, colors, skips, steps, window, ax)

ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("reward")
#ax.set_ylim(-410, 0)
fig.tight_layout()
plt.savefig("test")