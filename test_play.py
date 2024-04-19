import gymnasium as gym
from gymnasium.utils import play

from pathological_mc import PathologicalMountainCarEnv

env = play.play(gym.make('PathologicalMountainCar-v1', render_mode='rgb_array', terminate=True).env,
                zoom=1,
                keys_to_action={"1":0, "a":0, "2":2, "d":2,},
                noop=1)