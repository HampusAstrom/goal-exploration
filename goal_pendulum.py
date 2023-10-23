__credits__ = ["Hampus Åström"] 

from typing import Optional
from collections import OrderedDict

import numpy as np

from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv
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

class GoalPendulumEnv(PendulumEnv):
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 reward_type: str = "dense",
                 g=10.0,
                 fixed_goal: np.ndarray = None):
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
        self.reward_type = reward_type
        self.goal = None
        # TODO verify input of fixed goal
        self.fixed_goal = fixed_goal

    def _get_obs(self):
        theta, thetadot = self.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        goal = self.goal
        return OrderedDict([
            ("observation", obs),
            ("achieved_goal", obs),
            ("desired_goal", goal),
        ])
    
    def reset(self, *, 
              seed: Optional[int] = None, 
              options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
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
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        # the base environment seems to have an error as it does not
        # trunkate after 200 steps
        return obs, reward, term, trunk, info

    def compute_reward(self, 
                       achieved_goal: np.ndarray, 
                       desired_goal: np.ndarray, info) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            return (distance <= 0.45).astype(np.float64)