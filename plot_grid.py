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

class GridPlotter:
    def __init__(self,
                 model,
                 base_env,
                 goal_env,
                 size=10000,
                 flip_dims = False,
                 flip_y = False,
                 obs_grid_converter = None,
                 ) -> None:
        self.model = model
        self.base_env = base_env
        self.goal_env = goal_env
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
            self.high = self.goal_env.observation_space["observation"].high
            self.low = self.goal_env.observation_space["observation"].low
            # TODO handle non-closed dims
            self.cell_size = (self.high-self.low)/self.shape
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


    def sample_in_cell(self, cell):
        # select a goal in this cell
        rand = np.random.rand(len(self.shape))
        return self.low + (cell+rand)*self.cell_size

    def center_of_cell(self, cell):
        scell = tuple(np.array(cell) + 0.5)
        return self.low + (scell)*self.cell_size

    def point2cell(self, obs):
        # obs = obs["observation"]
        cell = tuple(np.floor((obs-self.low)/self.cell_size).astype(int))
        # TODO replace with nicer solution
        ret = list(cell)
        for i, v in enumerate(cell):
            ret[i] = min(max(cell[i], 0), self.shape[i]-1)
        #cell = min(cell, )
        return tuple(ret)

    def get_q_vals(self, goal_s=None):
        #ret = np.zeros(shape=(self.shape), dtype=int)
        # TODO hardcoded version for discrete actionspaces
        ret_shape = self.shape.copy()
        ret_shape.append(self.goal_env.action_space.n)
        ret = np.zeros(shape=ret_shape)

        if goal_s is None:
            goal = np.array([-1.65, -0.02,]) # hardcoded for patho MC
        else:
            goal = np.fromstring(goal_s, dtype=float, sep=' ')

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
        cmap.set_under(color='white')

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
                           cmap="viridis")
            ax = fig.add_subplot(subplot_rows, nplots, nplots+p+1)
            # TODO fix colormap scale, especially for segement version, so they match
            plt.pcolormesh(self.obss[..., self.first],
                        self.obss[..., self.second]*self.flip_y,
                        vals[..., p]-medians,
                        cmap="plasma")
            if segment:
                ax = fig.add_subplot(subplot_rows, nplots, 2*nplots+p+1)
                plt.pcolormesh(self.obss[..., self.first],
                            self.obss[..., self.second]*self.flip_y,
                            filtered_over_median[..., p],
                            cmap=cmap,
                            vmin=0.0000001)

        #plt.show()

    def plot_transitions(self, vals):
        # TODO names of axes and subfigs

        nplots = vals.shape[-2]

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
        cmap = mpl.cm.get_cmap("plasma").copy()
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
                        vals[..., p, self.first],
                        vals[..., p, self.second]*self.flip_y,
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
            # ax = fig.add_subplot(1, nplots, p+1, projection='3d')
            # surf = ax.plot_surface(self.obss[..., 0],
            #                        self.obss[..., 1],
            #                        vals[..., p],
            #                        cmap="viridis")
            #ax = fig.add_subplot(subplot_rows, nplots, p+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.quiver(self.obss[..., self.first],
                        self.obss[..., self.second]*self.flip_y,
                        diff[..., p, self.first],
                        diff[..., p, self.second]*self.flip_y,
                        angles='xy',
                        scale_units='xy',
                        scale=1,
                        color=color[p],
                        alpha=0.9)
            # plt.scatter(end[..., p, 1],
            #             end[..., p, 0],
            #             color=color[p],
            #             marker='.',
            #             alpha=0.4)
            # plt.scatter(self.obss[..., 1],
            #             self.obss[..., 0],
            #             color='k',
            #             marker='.',
            #             alpha=0.4)

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

        #plt.show()

    def plot_goal_data(self, folder):
        files = []
        grid_files = ["n_last_visit", "n_latest_succeded", "n_latest_targeted",
                      "n_succeded", "n_targeted", "n_visits", ]
        time_files = ["goals", "initial_targeted_goals",
                      "successful_goal_index", "successful_goal_spread"]
        #"n_shape",

        px = 1/plt.rcParams['figure.dpi']
        #fig = plt.figure(figsize=(1920*px, 1080*px))
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

        cmap = mpl.cm.get_cmap("plasma").copy()
        cmap.set_under(color='white')
        for p, measure in enumerate(grid_files):
            path = os.path.join(folder, measure)

            with open(path,'r') as measure_info:
                reader = csv.reader(measure_info, delimiter=' ')
                data = np.array([float(i[0]) for i in reader]).reshape(4,4)
                # x = list(reader)
                # data = np.array(x).astype("float")
                #data = np.array([float(i[0]) for i in reader]).reshape(100,100)

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
            im = plt.pcolormesh(self.obss[..., self.first],
                        self.obss[..., self.second]*self.flip_y,
                        data,
                        cmap="viridis")
            fig.colorbar(im)
            ax.set_title(measure)

        #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1/2000000steps_grid-novelty-baseline_256-256-256-256-256net/exp1")
    parser.add_argument('-g', '--goal', default=None)
    parser.add_argument('-s', '--segment', action="store_true")
    parser.add_argument('-c', '--cells', default=10000, type=int)
    args = parser.parse_args()

    #base_env = gym.make("PathologicalMountainCar-v1.1") # maybe we need some extra params...
    base_env = gym.make("FrozenLake-v1", is_slippery=False) # maybe we need some extra params...
    goal_env = GoalWrapper(base_env,
                                    goal_weight=1.0,
                                    goal_range=0.1,
                                    reward_func="term")

    model_name = os.path.join(args.name, "model")
    print(model_name)
    model = DQN.load(model_name, env=goal_env)
    model.policy.set_training_mode(False)
    print(model.policy)

    gp = GridPlotter(model,
                     base_env,
                     goal_env,
                     args.cells,
                     flip_dims=False,
                     flip_y=True,
                     obs_grid_converter=frozen_obs2grid)
    # print(f"Goal: {args.goal}")

    q_vals = gp.get_q_vals(goal_s=args.goal)
    # print(q_vals)
    # print(f"Max q: {np.max(q_vals)}")
    # print(f"Min q: {np.min(q_vals)}")
    gp.plot_q_vals(q_vals, args.segment)

    end, diff = gp.get_steps()
    #print(diff)
    #print(diff.shape)

    gp.plot_goal_data(args.name)

    gp.plot_transitions(diff)
    gp.plot_transitions_single(end, diff)

    gp.plot_selected_transitions(q_vals, diff)

    plt.show()

    #load_n_plot_q_vals(name=args.name)