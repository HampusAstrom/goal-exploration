import numpy as np
import matplotlib.pyplot as plt
import os

paths = ["./output/pendulum/trunc_pendulum_eval/fixed_dense_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_dense_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_sparse_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_sparse_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_truncated_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_truncated_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_dense_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_dense_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_sparse_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_sparse_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_truncated_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_truncated_pendulum_10000/exp1/eval_logs",
        ]
labels = ["fixed_dense_genericnormed_10000",
          "fixed_dense_pendulum_10000",
          "fixed_sparse_genericnormed_10000",
          "fixed_sparse_pendulum_10000",
          "fixed_truncated_genericnormed_10000",
          "fixed_truncated_pendulum_10000",
          "random_dense_genericnormed_10000",
          "random_dense_pendulum_10000",
          "random_sparse_genericnormed_10000",
          "random_sparse_pendulum_10000",
          "random_truncated_genericnormed_10000",
          "random_truncated_pendulum_10000",]

# current selection
paths = ["./output/pendulum/trunc_pendulum_eval/fixed_dense_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_dense_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_truncated_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_truncated_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_dense_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_dense_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_truncated_genericnormed_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/random_truncated_pendulum_10000/exp1/eval_logs",
        ]
labels = ["fixed_dense_genericnormed_10000",
          "fixed_dense_pendulum_10000",
          "fixed_truncated_genericnormed_10000",
          "fixed_truncated_pendulum_10000",
          "random_dense_genericnormed_10000",
          "random_dense_pendulum_10000",
          "random_truncated_genericnormed_10000",
          "random_truncated_pendulum_10000",]

linestyles = ["r-",
             "r--",
             "r:",
             "r-.",
             "b-",
             "b--",
             "b:",
             "b-.",
             ]

px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
window = 10
for path, label, linestyle in zip(paths, labels, linestyles):
    data = np.loadtxt(os.path.join(path, "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
    avg_data = np.convolve(data, [1]*window, 'valid')
    x = range(len(avg_data))
    ax.plot(x, avg_data, linestyle, label=label)

ax.legend()
plt.savefig("test_plot")

paths = ["./output/pendulum/dense_pendulum_eval/random_dense_pendulum_10000/exp1/eval_logs",
        "./output/pendulum/trunc_pendulum_eval/fixed_dense_pendulum_10000/exp1/eval_logs",
        ]

labels = ["random_dense_pendulum_10000",
          "fixed_dense_pendulum_10000",
          ]