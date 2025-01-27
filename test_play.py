import gymnasium as gym
from gymnasium.utils import play

from pathological_mc import PathologicalMountainCarEnv


def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
    return (obs_tp[0], obs_tp[1])

plotter = play.PlayPlot(compute_metrics, horizon_timesteps=200,
                   plot_names=["Position", "Velocity"])

if False:
    callback = plotter.callback
else:
    callback = None

env = play.play(gym.make('PathologicalMountainCar-v1.1',
                         render_mode='rgb_array',
                         terminate=True).env,
                zoom=2,
                keys_to_action={"1":0, "a":0, "2":2, "d":2,},
                noop=1,
                callback=callback)
