import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product
import inspect
import json
import utils
import csv
import argparse
import torch as th

from stable_baselines3 import SAC, HerReplayBuffer, DQN
from stable_baselines3.dqn import DQNwithICM
from stable_baselines3.sac import SACwithICM
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
#from stable_baselines3.common.utils import obs_as_tensor

#import imageio

import time
import yappi

from goal_wrapper import GoalWrapper, FiveXGoalSelection, OrderedGoalSelection, GridNoveltySelection

from sparse_pendulum import SparsePendulumEnv
from pathological_mc import PathologicalMountainCarEnv

def frozen_obs2grid(obs, grid = [4, 4]):
    x = obs % grid[0]
    y = obs // grid[1]
    return [x, y]

def frozen_grid2obs(grid_cell, grid = [4, 4]):
    return grid_cell[1]*grid[0] + grid_cell[0]


cdict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
}

custom1 = mpl.colors.LinearSegmentedColormap('BlueRed1', cdict)


class GridPlotter:
    def __init__(self,
                 model,
                 base_env,
                 goal_env,
                 goal,
                 size=10000,
                 size_goals=10000,
                 flip_dims = False,
                 flip_y = False,
                 obs_grid_converter = None,
                 ) -> None:
        self.model = model
        self.base_env = base_env
        self.goal_env = goal_env
        self.goal = goal
        self.size = size
        self.flipdim = flip_dims
        self.obs_grid_converter = obs_grid_converter
        if flip_y:
            self.flip_y = -1
        else:
            self.flip_y = 1

        if self.flipdim:
            self.first = 1
            self.second = 0
        else:
            self.first = 0
            self.second = 1

        dims = gym.spaces.utils.flatdim(self.goal_env.observation_space["observation"])

        if not obs_grid_converter:
            # TODO handle if this needs flattening
            self.shape = [int(size**(1/dims))]*dims
            self.shape_goals = [int(size_goals**(1/dims))]*dims
            self.high = self.goal_env.observation_space["observation"].high
            self.low = self.goal_env.observation_space["observation"].low
            # TODO handle non-closed dims
            self.cell_size = (self.high-self.low)/self.shape
            self.cell_size_goals = (self.high-self.low)/self.shape_goals
        else:
            self.obs_grid_converter = obs_grid_converter
            self.shape = [4, 4] # TODO replace shitty hardcoded

        # TODO this could get very large, do on the fly in that case
        obs_arr = []
        for index in np.ndindex(tuple(self.shape)):
            if not obs_grid_converter:
                obs_arr.append(self.center_of_cell(index))
            else:
                obs_arr.append(index)
        obss = np.array(obs_arr)
        self.obss = obss.reshape(self.shape+[len(self.shape)])

        obs_arr = []
        for index in np.ndindex(tuple(self.shape_goals)):
            if not obs_grid_converter:
                obs_arr.append(self.center_of_cell(index, self.cell_size_goals))
            else:
                obs_arr.append(index)
        obss = np.array(obs_arr)
        self.obss_goals = obss.reshape(self.shape_goals+[len(self.shape_goals)])

    def sample_in_cell(self, cell):
        # select a goal in this cell
        rand = np.random.rand(len(self.shape))
        return self.low + (cell+rand)*self.cell_size

    def center_of_cell(self, cell, cell_size=None):
        if cell_size is None:
            cell_size = self.cell_size
        scell = tuple(np.array(cell) + 0.5)
        return self.low + (scell)*cell_size

    def point2cell(self, obs):
        # obs = obs["observation"]
        cell = tuple(np.floor((obs-self.low)/self.cell_size).astype(int))
        # TODO replace with nicer solution
        ret = list(cell)
        for i, v in enumerate(cell):
            ret[i] = min(max(cell[i], 0), self.shape[i]-1)
        #cell = min(cell, )
        return tuple(ret)

    def show_goal_point(self):
        if self.obs_grid_converter:
            goal = self.obs_grid_converter(self.goal)
        else:
            goal = self.goal
        plt.scatter(goal[0], goal[1], s=200, color='k', marker='*')
        plt.scatter(goal[0], goal[1], s=100, color='gold', marker='*')

    def get_q_vals(self, goal):
        #ret = np.zeros(shape=(self.shape), dtype=int)
        # TODO hardcoded version for discrete actionspaces
        ret_shape = self.shape.copy()
        ret_shape.append(self.goal_env.action_space.n)
        ret = np.zeros(shape=ret_shape)

        for index in np.ndindex(tuple(self.shape)):
            obs = self.obss[index]
            if self.obs_grid_converter:
                obs = frozen_grid2obs(obs)
                if len(goal) > 1:
                    goal = frozen_grid2obs(goal)
            observation = {"observation": obs,
                           "achieved_goal": obs,
                           "desired_goal": goal}
            #action, _ = self.model.q_net.predict(observation, deterministic=True)
            #print(action)
            observation, _ = self.model.policy.obs_to_tensor(observation)
            #print(observation)
            with th.no_grad():
                # extracted_obs = self.model.q_net.extract_features(observation,
                #                                                   self.model.q_net.features_extractor)
                #print(observation)
                q_values = self.model.q_net.forward(observation).cpu().numpy()
            #r = max(q_values)
            ret[index] = q_values[0]

        return ret

    def get_steps(self):
        ret_shape = self.shape.copy()
        ret_shape.append(self.goal_env.action_space.n)
        ret_shape.append(2) # TODO replace hardcoded
        diff = np.zeros(shape=ret_shape)
        end = np.zeros(shape=ret_shape)

        for index in np.ndindex(tuple(self.shape)):
            start_obs = self.obss[index]
            if self.obs_grid_converter:
                start_obs = frozen_grid2obs(start_obs)
            end_obs = []
            for a in range(self.goal_env.action_space.n):
                # TODO OBS specific for patho MC, setting state will be different in each env
                self.base_env.reset()
                if self.obs_grid_converter:
                    self.base_env.unwrapped.s = start_obs
                else:
                    self.base_env.unwrapped.state = start_obs #(x, v)
                obs, _, _, _, _, = self.base_env.step(a)
                end_obs.append(obs)
            if self.obs_grid_converter:
                end_row = []
                diff_row = []
                for a in range(self.goal_env.action_space.n):
                    end_row.append(self.obs_grid_converter(end_obs[a]))
                    diff_row.append(np.array(end_row[a]) - np.array(self.obs_grid_converter(start_obs)))
                end[index] = end_row
                diff[index] = diff_row
            else:
                end[index] = end_obs
                diff[index] = end_obs - start_obs

        return end, diff

    def plot_q_vals(self, vals, segment=False):
        # TODO names of axes and subfigs

        nplots = vals.shape[-1]
        means = np.mean(vals, -1)
        medians = np.median(vals, -1) # for three values median is the next highest value
        maxes = np.max(vals, -1)
        max_filter = vals < maxes[..., np.newaxis]
        filtered_over_median = vals-medians[..., np.newaxis]
        filtered_over_median[max_filter] = -1
        advantage = vals - maxes[..., np.newaxis]
        filtered_advantage = advantage.copy()
        filtered_advantage[vals == maxes[..., np.newaxis]] = 1

        if segment:
            subplot_rows = 3
        else:
            subplot_rows = 2

        px = 1/plt.rcParams['figure.dpi']
        #fig = plt.figure(figsize=(1920*px, 1080*px))
        fig, axes = plt.subplots(subplot_rows, nplots, layout='constrained',
                                 figsize=(1920*px, 1920/2*px),)
        # shitty override that I don't have the energy to do in a nicer way atm
        for ax2 in axes:
            for ax in ax2:
                ax.set_xticks([])
                ax.set_yticks([])

        #x, y = zip(*self.obss.copy().reshape(-1, 2))
        #valrow = vals.reshape(-1, nplots)
        cmap = mpl.cm.get_cmap("plasma").copy()
        #cmap.set_under(color='white')
        cmap.set_over(color='white')
        custom2 = custom1.copy()
        custom2.set_over(color='black')
        #q_norm = mpl.colors.CenteredNorm(vcenter=0, clip=True)
        #q_norm.vmax = 1

        for p in range(nplots):
            # ax = fig.add_subplot(1, nplots, p+1, projection='3d')
            # surf = ax.plot_surface(self.obss[..., 0],
            #                        self.obss[..., 1],
            #                        vals[..., p],
            #                        cmap="viridis")
            ax = fig.add_subplot(subplot_rows, nplots, p+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.pcolormesh(self.obss[..., self.first],
                           self.obss[..., self.second]*self.flip_y,
                           vals[..., p],
                           norm=mpl.colors.CenteredNorm(),
                           cmap=custom2)
            plt.colorbar()
            self.show_goal_point()
            ax = fig.add_subplot(subplot_rows, nplots, nplots+p+1)
            # TODO fix colormap scale, especially for segement version, so they match
            plt.pcolormesh(self.obss[..., self.first], # show distance between selected action and next best
                        self.obss[..., self.second]*self.flip_y,
                        vals[..., p]-medians,
                           norm=mpl.colors.CenteredNorm(),
                        cmap=custom1)
            plt.colorbar()
            self.show_goal_point()
            # if segment and np.max(filtered_over_median[..., p])>np.min(filtered_over_median[..., p]):
            #     ax = fig.add_subplot(subplot_rows, nplots, 2*nplots+p+1)
            #     plt.pcolormesh(self.obss[..., self.first],
            #                 self.obss[..., self.second]*self.flip_y,
            #                 filtered_over_median[..., p],
            #                 cmap=cmap,
            #                 vmin=0.0000001)
            if segment and np.max(filtered_advantage[..., p])>np.min(filtered_advantage[..., p]):
                ax = fig.add_subplot(subplot_rows, nplots, 2*nplots+p+1)
                plt.pcolormesh(self.obss[..., self.first],
                            self.obss[..., self.second]*self.flip_y,
                            filtered_advantage[..., p],
                            cmap=cmap,
                            vmax=0.0000001,
                            )
            plt.colorbar()
            self.show_goal_point()

        #plt.show()

    def plot_transitions(self, q_diffs):
        # TODO names of axes and subfigs

        nplots = q_diffs.shape[-2]

        subplot_rows = 1

        px = 1/plt.rcParams['figure.dpi']
        #fig = plt.figure(figsize=(1920*px, 1080*px))
        fig, axes = plt.subplots(subplot_rows, nplots, layout='constrained',
                                figsize=(1920*px, 1920/3*px),)
        # shitty override that I don't have the energy to do in a nicer way atm
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        #x, y = zip(*self.obss.copy().reshape(-1, 2))
        #valrow = vals.reshape(-1, nplots)
        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_under(color='white')

        if self.obs_grid_converter:
            color = ["red", "green", "blue", "black"]
        else:
            color = ["red", "green", "blue"]

        for p in range(nplots):
            # ax = fig.add_subplot(1, nplots, p+1, projection='3d')
            # surf = ax.plot_surface(self.obss[..., 0],
            #                        self.obss[..., 1],
            #                        vals[..., p],
            #                        cmap="viridis")
            ax = fig.add_subplot(subplot_rows, nplots, p+1)
            plt.quiver(self.obss[..., self.first],
                        self.obss[..., self.second]*self.flip_y,
                        q_diffs[..., p, self.first],
                        q_diffs[..., p, self.second]*self.flip_y,
                        angles='xy',
                        color=color[p])

        #plt.show()

    def plot_transitions_single(self, end, diff):
        # TODO names of axes and subfigs

        nplots = diff.shape[-2]

        subplot_rows = 1

        px = 1/plt.rcParams['figure.dpi']
        #fig = plt.figure(figsize=(1920*px, 1080*px))
        fig, ax = plt.subplots(1, 1, layout='constrained',
                                figsize=(1080*px, 1080*px),)
        # shitty override that I don't have the energy to do in a nicer way atm
        # ax.set_xticks([])
        # ax.set_yticks([])

        #x, y = zip(*self.obss.copy().reshape(-1, 2))
        #valrow = vals.reshape(-1, nplots)
        # cmap = mpl.cm.get_cmap("plasma").copy()
        # cmap.set_under(color='white')
        if self.obs_grid_converter:
            color = ["red", "green", "blue", "black"]
        else:
            color = ["red", "green", "blue"]

        for p in range(nplots):
            plt.quiver(self.obss[..., self.first],
                        self.obss[..., self.second]*self.flip_y,
                        diff[..., p, self.first],
                        diff[..., p, self.second]*self.flip_y,
                        angles='xy',
                        scale_units='xy',
                        scale=1,
                        color=color[p],
                        alpha=0.9)

        #plt.show()


    def plot_selected_transitions(self, q_vals, diffs):
        # TODO names of axes and subfigs

        nplots = diffs.shape[-2]

        subplot_rows = 1

        px = 1/plt.rcParams['figure.dpi']
        #fig = plt.figure(figsize=(1920*px, 1080*px))
        fig, ax = plt.subplots(1, 1, layout='constrained',
                                figsize=(1080*px, 1080*px),)

        if self.obs_grid_converter:
            color = ["red", "green", "blue", "black"]
        else:
            color = ["red", "green", "blue"]
        color_grid = np.full(shape=q_vals.shape, fill_value=color, dtype=str)
        maxes = np.max(q_vals, -1,keepdims=True)
        sel_ind = np.where(q_vals[:,:,] == maxes)
        max_ind = np.argmax(q_vals, -1, keepdims=True)
        plt.quiver(self.obss[..., self.first],
                    self.obss[..., self.second]*self.flip_y,
                    diffs[sel_ind][..., self.first],
                    diffs[sel_ind][..., self.second]*self.flip_y,
                    angles='xy',
                    scale_units='xy',
                    scale=1,
                    color=color_grid[sel_ind],
                    alpha=0.9)
        self.show_goal_point()

        #plt.show()

    def plot_goal_data(self, folder):
        files = []
        grid_files = ["n_last_visit", "n_latest_succeded", "n_latest_targeted",
                      "n_succeded", "n_targeted", "n_visits", ]
        time_files = ["goals", "initial_targeted_goals",
                      "successful_goal_index", "successful_goal_spread"]
        #"n_shape",

        px = 1/plt.rcParams['figure.dpi']
        # fig = plt.figure(figsize=(1920*px, 1080*px))
        subplot_rows = int(np.floor(np.sqrt(len(grid_files))))
        nplots = len(grid_files)
        fig, axes = plt.subplots(subplot_rows,
                                 nplots//subplot_rows,
                                 layout='constrained',
                                 figsize=(1920*px, 1920/2*px),)
        # shitty override that I don't have the energy to do in a nicer way atm
        for ax2 in axes:
            for ax in ax2:
                ax.set_xticks([])
                ax.set_yticks([])

        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_under(color='white')
        for p, measure in enumerate(grid_files):
            path = os.path.join(folder, measure)

            if not os.path.isfile(path):
                continue

            with open(path,'r') as measure_info:
                reader = csv.reader(measure_info, delimiter=' ')
                if self.obs_grid_converter is not None:
                    data = np.array([float(i[0]) for i in reader]).reshape(4,4)
                else:
                    x = list(reader)
                    data = np.array(x).astype("float")

            print(data)

            # ax = fig.add_subplot(1, nplots, p+1, projection='3d')
            # surf = ax.plot_surface(self.obss[..., 0],
            #                        self.obss[..., 1],
            #                        vals[..., p],
            #                        cmap="viridis")
            row = np.floor(p/subplot_rows)
            ax = fig.add_subplot(subplot_rows, nplots//subplot_rows, p+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            im = plt.pcolormesh(self.obss_goals[..., self.first],
                        self.obss_goals[..., self.second]*self.flip_y,
                        data,
                        cmap=cmap,
                        vmin=0.0000001)
            fig.colorbar(im)
            ax.set_title(measure)

        #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1/10000000steps_1.0goalrewardWeight_0.0fixedGoalFraction_0.1goal_range_term-reward_func_10000grid_size_0.1fraction_random_2dist_decay_her*0.5t2g_batch_size512_lr_1e-3_4x256resblock-x4net/exp2")
    parser.add_argument('-g', '--goal', default=None)
    parser.add_argument('-s', '--segment', action="store_true")
    parser.add_argument('-c', '--cells', default=10000, type=int)
    parser.add_argument('-o', '--goalcells', default=10000, type=int)
    args = parser.parse_args()

    if "Frozen" in args.name:
        obs_grid_converter = frozen_obs2grid
        base_env = gym.make("FrozenLake-v1", is_slippery=False) # maybe we need some extra params...
        reward_func = "exact_goal_match_reward"
    else:
        obs_grid_converter = None
        base_env = gym.make("PathologicalMountainCar-v1.1") # maybe we need some extra params...
        reward_func = "term"

    goal_env = GoalWrapper(base_env,
                                    goal_weight=1.0,
                                    goal_range=0.1,
                                    reward_func=reward_func)

    for file in os.listdir(args.name):
        if file.endswith(".zip"):
            file_name = file
            print(file_name)

    model_name = os.path.join(args.name, file_name)
    print(model_name)
    model = DQN.load(model_name, env=goal_env)
    model.policy.set_training_mode(False)
    print(model.policy)

    if args.goal is None:
        goal = np.array([-1.60, 0.00,]) # hardcoded for patho MC
    else:
        goal = np.fromstring(args.goal, dtype=float, sep=' ')

    gp = GridPlotter(model,
                     base_env,
                     goal_env,
                     goal = goal,
                     size=args.cells,
                     size_goals=args.goalcells,
                     flip_dims=False,
                     flip_y=False,
                     obs_grid_converter=obs_grid_converter)
    # print(f"Goal: {args.goal}")

    q_vals = gp.get_q_vals(goal=goal)
    # print(q_vals)
    print(f"Max q: {np.max(q_vals)}")
    print(f"Min q: {np.min(q_vals)}")
    gp.plot_q_vals(q_vals, args.segment)

    end, diff = gp.get_steps()
    #print(diff)
    #print(diff.shape)

    if "uniform" not in args.name:
        gp.plot_goal_data(args.name)

    gp.plot_transitions(diff)
    gp.plot_transitions_single(end, diff)

    gp.plot_selected_transitions(q_vals, diff)

    plt.show()