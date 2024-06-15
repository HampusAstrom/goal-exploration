import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os
from cycler import cycler

import torch as th
from typing import Any, Callable, Dict, Optional, Tuple, Union
from gymnasium.vector import VectorEnv
import gymnasium as gym

GymObs = Union[th.Tensor, Dict[str, th.Tensor]]

# TODO possibly use jax again?
def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def plot_targeted_goals(goals, coord_names, path):
    assert len(coord_names) == len(goals[1,:])
    order = np.linspace(0, 1, len(goals))

    fig = plt.figure(figsize=(2*len(coord_names), 2*len(coord_names)))

    for i in range(len(coord_names)**2):
        col = i % (len(coord_names)-1)
        row = i // (len(coord_names)-1)
        if col < row:
            continue # don't plot diagonal
        ax = fig.add_subplot(len(coord_names)-1, len(coord_names)-1, i+1)
        im = ax.scatter(goals[:,col+1], goals[:,row], c=order)
        if row == 0:
            ax.set_xlabel(coord_names[col+1])
        else:
            ax.set_xticks([])
        if col == len(coord_names)-2:
            ax.set_ylabel(coord_names[row])
        else:
            ax.set_yticks([])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    cbar_ax = fig.add_axes([0, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #plt.tight_layout()
    plt.savefig(os.path.join(path,"goal_spread"))
    plt.close(fig)

def get_all_folders(dir):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

    # for obj in subfolders:
    #     print(obj)
    return sorted(subfolders)

def add_subplot(path, window, ax):
    datas = []
    experiments = get_all_folders(path)
    for exp in experiments:
        data = np.loadtxt(os.path.join(exp, "eval_logs", "monitor.csv"), delimiter=',', skiprows=2, usecols=0)
        #avg_data = np.convolve(data, [1]*window, 'valid')/window
        datas.append(data)
    #avg_data = np.mean(datas, axis=0)
    #avg_std = np.std(datas, axis=0)
    data = np.mean(datas, axis=0)
    std = np.std(datas, axis=0)
    avg_data = np.convolve(data, [1]*window, 'valid')/window
    avg_std = np.convolve(std, [1]*window, 'valid')/window
    #x = range(len(avg_data))
    x = np.linspace(0, len(avg_data)*100, len(avg_data))
    ax.plot(x, avg_data, label=os.path.basename(path))
    ax.fill_between(x, avg_data+avg_std, avg_data-avg_std, alpha=0.03,)

def plot_all_in_folder(dir):
    subfolders = get_all_folders(dir)

    # TODO replace with something that adaps to number of configurations
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                           cycler('linestyle', ['-', ':',])))# '--', '-.'])))

    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    window = 1000

    for folder in subfolders:
        add_subplot(folder, window, ax)

    ax.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("steps")
    ax.set_ylabel("reward")
    #ax.set_ylim(-100, 200)
    fig.tight_layout()
    plt.savefig(os.path.join(dir, "eval_results"))

    #coord_names = ["x", "y", "ang. vel."]
    coord_names = ["xpos", "velocity"]
    for folder in subfolders:
        experiments = get_all_folders(folder)
        for exp in experiments:
            goal_file = os.path.join(exp, "goals")
            if os.path.isfile(goal_file):
                goals = np.loadtxt(goal_file, delimiter=' ')#, skiprows=2, usecols=0)
                plot_targeted_goals(goals, coord_names,exp)

# copied and altered from rllte
class Gymnasium2Torch(gym.Wrapper):
    """Env wrapper for processing gymnasium environments and outputting torch tensors.

    Args:
        env (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        envpool (bool): Whether to use `EnvPool` env.

    Returns:
        Gymnasium2Torch wrapper.
    """

    def __init__(self, env: VectorEnv, device: str, envpool: bool = False) -> None:
        super().__init__(env)
        #self.num_envs = env.unwrapped.num_envs # TODO solve nicer for all cases
        self.device = th.device(device)

        # envpool's observation space and action space are the same as the single env.
        if not envpool:
            #self.observation_space = env.single_observation_space
            #self.action_space = env.single_action_space
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        if isinstance(self.observation_space, gym.spaces.Dict):
            self._format_obs = lambda x: {key: th.as_tensor(item, device=self.device) for key, item in x.items()}
        else:
            self._format_obs = lambda x: th.as_tensor(x, device=self.device)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[GymObs, Dict]:
        """Reset all environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            First observations and info.
        """
        obs, infos = self.env.reset(seed=seed, options=options)

        return self._format_obs(obs), infos

    def step(self, actions: th.Tensor) -> Tuple[GymObs, th.Tensor, th.Tensor, th.Tensor, Dict[str, Any]]:
        """Take an action for each environment.

        Args:
            actions (th.Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            Next observations, rewards, terminateds, truncateds, infos.
        """
        new_observations, rewards, terminateds, truncateds, infos = self.env.step(actions.cpu().numpy())
        # TODO: get real next observations
        # for idx, (term, trunc) in enumerate(zip(terminateds, truncateds)):
        #     if term or trunc:
        #         new_obs[idx] = info['final_observation'][idx]

        # convert to tensor
        rewards = th.as_tensor(rewards, dtype=th.float32, device=self.device)

        terminateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in terminateds],
            dtype=th.float32,
            device=self.device,
        )
        truncateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in truncateds],
            dtype=th.float32,
            device=self.device,
        )

        return self._format_obs(new_observations), rewards, terminateds, truncateds, infos


if __name__ == '__main__':
    #plot_all_in_folder("./output/wrapper/SparsePendulumEnv-v1") #
    plot_all_in_folder("./output/wrapper/PathologicalMountainCar-v1.1")