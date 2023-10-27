__credits__ = ["Hampus Åström"] 

from typing import Optional
from collections import OrderedDict

import numpy as np

from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.envs.classic_control import utils

from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="GoalPendulumEnv-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="goal_pendulum:GoalPendulumEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)

DEFAULT_THETA = np.pi
DEFAULT_THETADOT = 1.0

class GoalPendulumEnv(PendulumEnv):
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 reward_type: str = "generic",
                 reward_density: str = "dense",
                 g=10.0,
                 fixed_goal: np.ndarray = None,
                 harder_start: float = None):
        super().__init__(
            render_mode=render_mode,
            g=g,
        )
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=-high, high=high, dtype=np.float32),
                achieved_goal=spaces.Box(low=-high, high=high, dtype=np.float32),
                observation=spaces.Box(low=-high, high=high, dtype=np.float32),
        ))
        reward_type = reward_type.lower()
        assert reward_type in {"generic", "genericnormed", "pendulum"}
        self.reward_type = reward_type

        reward_density = reward_density.lower()
        assert reward_density in {"dense", "sparse", "truncated"}
        self.reward_density = reward_density
        
        self.goal = None
        # TODO verify input of fixed goal
        self.fixed_goal = fixed_goal

        self.obs_norm = 2*high
        self.harder_start = harder_start
        # sets starting angle to be the in the "harder_start" fraction of the state space
        # that is furthest away from the goal state at 0

    def _get_obs(self):
        theta, thetadot = self.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        goal = self.goal
        #return OrderedDict([
        #    ("observation", obs),
        #    ("achieved_goal", obs),
        #    ("desired_goal", goal),
        #])
        return {"observation": obs, "achieved_goal": obs, "desired_goal": goal}
    
    def reset(self, *, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        if self.harder_start:
            range = min(self.harder_start, 1)
            high = range*DEFAULT_THETA
            theta = self.np_random.uniform(low=0, high=high)
            theta += DEFAULT_THETA-range
            theta *= 1 if self.np_random.random() < 0.5 else -1
            self.state[0] = theta

        # for now we generate the goals uniformly from the state space
        if self.fixed_goal is None:
            self.goal = self.observation_space.sample()["desired_goal"]
        else:
            self.goal = self.fixed_goal
        return self._get_obs(), {}

    def step(self, u):
        obs, _, term, trunk, info = super().step(u)
        # for now we throw away the external reward and just reward
        # proximity to goal, either directly or sparsely
        # currently not normalizing for different scales
        info["u"] = u
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        # the base environment seems to have an error as it does not
        # trunkate after 200 steps
        return obs, reward, term, trunk, info

    def compute_reward(self, 
                       achieved_goal: np.ndarray, 
                       desired_goal: np.ndarray, info: dict) -> float:
        if self.reward_type == "generic" or self.reward_type == "genericnormed":
            if self.reward_type == "genericnormed":
                distance = np.linalg.norm((achieved_goal - desired_goal)/self.obs_norm, axis=-1)
                cutoff = 0.8
            else:
                distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
                cutoff = 0.5

            if self.reward_density == "dense":
                return np.exp(-distance)
            elif self.reward_density == "sparse":
                    return (distance <= 0.45).astype(np.float64)
            elif self.reward_density == "truncated":
                reward = np.exp(-distance)
                if reward.shape != ():
                    trunc = reward >= cutoff
                    reward[trunc] = 0
                    return -reward
                if reward < cutoff:
                    return 0
                else:
                    return reward
        elif self.reward_type == "pendulum":
            thetaA = self.get_angle(achieved_goal)
            thetaD = self.get_angle(desired_goal)
            relAngle = np.abs(thetaD-thetaA)
            relSpeed = np.abs(desired_goal[...,2]-achieved_goal[...,2])
            if not isinstance(info, dict):
                u = np.asarray([d['u'][0] for d in info])
            else:
                u = info["u"][0]
            cost = relAngle ** 2 + 0.1 * relSpeed**2 + 0.001 * (u**2)
            if self.reward_density == "dense":
                return -cost
            elif self.reward_density == "sparse":
                return (cost < 1).astype(np.float64)
                # this number is arbitraty, but cost can go from 0 to ~16.3
            elif self.reward_density == "truncated":
                if cost.shape != ():
                    trunc = cost >= 1
                    cost[trunc] = 2
                    return -cost
                else:
                    if cost < 1:
                        return -cost
                    else:
                        return -2
                # cost jump to platau to make rewards sparse but retain fine tuning
                
    def get_angle(self,obs):
        return angle_normalize(np.arctan2(obs[...,1], obs[...,0]))