from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np
from typing import Optional

REWARD_MODE_BINARY = 'binary' # return 1 when pendulum within allowed speed and angle limit over balance_counter time steps
REWARD_MODE_SPARSE = 'sparse' # return continuous speed reward only within speed and range limit

from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="SparsePendulumEnv-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="sparse_pendulum:SparsePendulumEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)

DEFAULT_THETA = np.pi
DEFAULT_THETADOT = 1.0

class SparsePendulumEnv(PendulumEnv):

    def __init__(self,
                 reward_angle_limit = np.pi * 0.05, # currently ignored if sparse
                 reward_speed_limit = 8, # currently ignored if sparse
                 balance_counter = 5,
                 reward_mode = REWARD_MODE_SPARSE,
                 harder_start: float = None):
        super().__init__()
        self.reward_speed_limit = reward_speed_limit
        self.reward_angle_limit = reward_angle_limit
        self.balance_counter = balance_counter
        self.reward_mode = reward_mode
        self.harder_start = harder_start

        self.current_balance_counter = 0

    def reward_binary(self, th, thdot, u):
        angle = angle_normalize(th)
        done = False
        reward = 0

        if self.check_angle_speed_limit(angle, thdot):
            self.current_balance_counter += 1
            if self.current_balance_counter >= self.balance_counter:
                self.current_balance_counter = 0
                reward = 1
                done = True

        return reward, done

    def reward_sparse(self, th, thdot, u):
        angle = angle_normalize(th)
        #done = False
        #reward = 0

        #if self.check_angle_speed_limit(angle, thdot):
            #reward = self.max_speed - (np.absolute(thdot) / 6.0)

        #return reward, done
        cost = angle ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        if cost.shape != ():
            trunc = cost >= 1
            cost[trunc] = 2
            return -cost, False
        else:
            if cost < 1:
                return -cost, False
            else:
                return -2, False

    def check_angle(self, angle):
        return (angle >= -self.reward_angle_limit) and (angle <= self.reward_angle_limit)

    def check_speed(self, thdot):
        return (thdot >= -self.reward_speed_limit) and (thdot <= self.reward_speed_limit)

    def check_angle_speed_limit(self, angle, thdot):
        return self.check_angle(angle) and self.check_speed(thdot)

    def reset(self, *,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        if self.harder_start:
            range = min(self.harder_start, 1)
            high = range*DEFAULT_THETA
            theta = np.random.uniform(low=0, high=high)
            theta += DEFAULT_THETA-range
            theta *= 1 if np.random.random() < 0.5 else -1
            self.state[0] = theta

        return self._get_obs(), {}

    def step(self, u):
        obs, _, term, trunk, info = super().step(u)

        u = self.last_u
        info["u"] = u
        th, thdot = self.state

        if self.reward_mode == REWARD_MODE_BINARY:
            reward, done = self.reward_binary(th, thdot, u)
        elif self.reward_mode == REWARD_MODE_SPARSE:
            reward, done = self.reward_sparse(th, thdot, u)

        return obs, reward, done, trunk, info
