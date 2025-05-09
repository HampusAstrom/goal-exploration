import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import inspect
import json
import utils
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


class GridPlotter:
    def __init__(self,
                 model,
                 base_env,
                 goal_env,
                 size=10000
                 ) -> None:
        self.model = model
        self.base_env = base_env
        self.goal_env = goal_env
        self.size = size

        dims = gym.spaces.utils.flatdim(self.goal_env.observation_space["observation"])
        self.high = self.goal_env.observation_space["observation"].high
        self.low = self.goal_env.observation_space["observation"].low

        # # TODO handle if this needs flattening
        self.shape = [int(size**(1/dims))]*dims
        # # TODO handle non-closed dims
        self.cell_size = (self.high-self.low)/self.shape

        # TODO this could get very large, do on the fly in that case
        obs_arr = []
        for index in np.ndindex(tuple(self.shape)):
            obs_arr.append(self.center_of_cell(index))
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
            goal = np.array([-1.65, -0.02,])
        else:
            goal = np.fromstring(goal_s, dtype=float, sep=' ')

        for index in np.ndindex(tuple(self.shape)):
            obs = self.obss[index]
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

    def plot_q_vals(self, vals, segment=False):
        # TODO names of axes and subfigs

        nplots = vals.shape[-1]
        means = np.mean(vals, -1)
        maxes = np.max(vals, -1)
        max_filter = vals < maxes[..., np.newaxis]
        filtered_over_mean = vals-means[..., np.newaxis]
        filtered_over_mean[max_filter] = -1

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

        for p in range(nplots):
            # ax = fig.add_subplot(1, nplots, p+1, projection='3d')
            # surf = ax.plot_surface(self.obss[..., 0],
            #                        self.obss[..., 1],
            #                        vals[..., p],
            #                        cmap="viridis")
            ax = fig.add_subplot(subplot_rows, nplots, p+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            plt.pcolormesh(self.obss[..., 1],
                           self.obss[..., 0],
                           vals[..., p],
                           cmap="viridis")
            ax = fig.add_subplot(subplot_rows, nplots, nplots+p+1)
            # TODO fix colormap scale, especially for segement version, so they match
            plt.pcolormesh(self.obss[..., 1],
                        self.obss[..., 0],
                        vals[..., p]-means,
                        cmap="RdYlGn")
            if segment:
                ax = fig.add_subplot(subplot_rows, nplots, 2*nplots+p+1)
                plt.pcolormesh(self.obss[..., 1],
                            self.obss[..., 0],
                            filtered_over_mean[..., p],
                            cmap="RdYlGn")

        plt.show()



    # def load_n_plot_q_vals(self, name):
        # algo_kwargs = {"replay_buffer_class": HerReplayBuffer,
        #                    "replay_buffer_kwargs": dict(
        #                    n_sampled_goal=4,
        #                    goal_selection_strategy="future",
        #                    copy_info_dict=True,), # TODO Turn off if not needed
        #                    }

        # model = DQN("MultiInputPolicy",
        #             train_env,
        #             **algo_kwargs,
        #             learning_starts=300,
        #             verbose=0,
        #             buffer_size=100000,
        #             learning_rate=1e-3,
        #             gamma=0.95,
        #             batch_size=256,
        #             policy_kwargs=dict(net_arch=[256, 256, 256],),
        #             #policy_kwargs=dict(net_arch=[64, 64],),
        #             #policy_kwargs=dict(net_arch=[256, 256, 256, 256, 256],),
        #             seed=None,
        #             device="cuda",
        #             tensorboard_log=None,
        # )

        # self.plot_q_vals(model, base_env, goal_env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1/2000000steps_grid-novelty-baseline_256-256-256-256-256net/exp1/model.zip")
    parser.add_argument('-g', '--goal', default=None)
    parser.add_argument('-s', '--segment', action="store_true")
    args = parser.parse_args()

    base_env = gym.make("PathologicalMountainCar-v1.1") # maybe we need some extra params...
    goal_env = GoalWrapper(base_env,
                                    goal_weight=1.0,
                                    goal_range=0.1,
                                    reward_func="reselect")

    model = DQN.load(args.name, env=goal_env)
    model.policy.set_training_mode(False)
    print(model.policy)

    gp = GridPlotter(model, base_env, goal_env, 10000)

    print(f"Goal: {args.goal}")

    q_vals = gp.get_q_vals(goal_s=args.goal)
    print(q_vals)
    print(f"Max q: {np.max(q_vals)}")
    print(f"Min q: {np.min(q_vals)}")
    gp.plot_q_vals(q_vals, args.segment)


    #load_n_plot_q_vals(name=args.name)