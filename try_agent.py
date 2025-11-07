import argparse

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from goal_wrapper import GoalWrapper, FiveXGoalSelection, OrderedGoalSelection

from stable_baselines3 import SAC, HerReplayBuffer, DQN
from stable_baselines3.dqn import DQNwithICM
from stable_baselines3.sac import SACwithICM

from sparse_pendulum import SparsePendulumEnv
from pathological_mc import PathologicalMountainCarEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="seeds 2,3 on the easeir side in path MC, seed 0 at the bottom, 4 on hard side")
    # parser.add_argument('-g', '--goals', nargs='*', default=[[-1.65, -0.02,]])
    #parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1/2000000steps_grid-novelty-baseline_256-256-256-256-256net/exp1/model.zip")
    parser.add_argument('-n', '--name', default="/home/hampus/rl/goal-exploration/output/wrapper/PathologicalMountainCar-v1.1/10000000steps_grid-novelty-baseline_batch_size512_train_freq512_lr_1e-4_5x256net/exp1/model.zip")
    parser.add_argument('-g', '--goal', default=None)
    parser.add_argument('-s', '--seed', type=int , default=None)
    args = parser.parse_args()

    num_eval_episodes = 4 # TODO replace with input option
    env_id = "PathologicalMountainCar-v1.1" # TODO replace with input option

    # env = gym.make("CartPole-v1", render_mode="rgb_array")  # replace with your environment
    env = gym.make(env_id, terminate=True, ) #render_mode="rgb_array") #, render_mode='human')
    if args.goal is None:
        goal = np.array([-1.65, -0.02,])
    else:
        goal = np.fromstring(args.goal, dtype=float, sep=' ')

    env = GoalWrapper(env,
                    goal_weight=1.0,
                    goal_range=0.1,
                    after_goal_success="local-reselect",
                    goal_selection_strategies = lambda obs: goal)
    # Add monitor wrapper? prob not?

    # env = RecordVideo(env, video_folder="test_video", name_prefix="eval",
    #                 episode_trigger=lambda x: True)
    #env = RecordEpisodeStatistics(env, deque_size=num_eval_episodes)

    # TODO add wrapper to store trajectories

    # load the saved movel
    model = DQNwithICM.load(args.name,
                            env=env)

    trajs = []

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset(seed=args.seed)
        print(obs)

        traj = [obs["observation"]]
        episode_over = False
        while not episode_over:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs["observation"])

            episode_over = terminated or truncated
        trajs.append(traj)
    env.close()

    #print(f'Episode time taken: {env.time_queue}')
    #print(f'Episode total rewards: {env.return_queue}')
    #print(f'Episode lengths: {env.length_queue}')

    # TODO plot trajectory

    #fig, ax = plt.subplots(2, 2)
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(1920*px, 1080*px))
    for i, traj in enumerate(trajs):
        ax = plt.subplot(int(np.ceil(np.sqrt(num_eval_episodes))),
                         int(np.ceil(np.sqrt(num_eval_episodes))),
                         i+1)
        pos, vel = zip(*traj)
        col = range(len(pos))
        ax.scatter(vel, pos, c=col, marker='.')
        ax.set_title(f"Length {len(pos)}")
        # add markers for goal areas
        ax.fill_between([0, 0.07], [0.63, 0.63], [0.5, 0.5],
                        alpha=0.5, fc="salmon", ec="red")
        ax.fill_between([-0.07, 0], [-1.6, -1.6], [-1.73, -1.73],
                        alpha=0.5, fc="gold", ec="goldenrod")
        ax.scatter(0, -0.473409, c="r", marker='x')
        ax.scatter(goal[1], goal[0], c="k", marker='x')
    plt.show()
