import numpy as np
import matplotlib.pyplot as plt
import csv

with open('/shared-folder/output/wrapper/SparsePendulumEnv-v1/20000steps_1.0goalrewardWeight_0.0fixedGoalFraction_[1,1,5,1,1]component_weights/exp1/eval_logs/goal_component_map.txt', newline='') as csvfile:

    reader = csv.reader(filter(lambda row: row[0]!='#', csvfile),
                        delimiter=' ',
                        quoting=csv.QUOTE_NONNUMERIC)

    data = np.array(list(reader))

    print("mean values of each column:")
    means = np.mean(data, 0)
    maxes = np.max(data, 0)
    np.set_printoptions(precision=5, suppress=True)
    print(means)
    print("max values of each column:")
    print(maxes)