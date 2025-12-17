import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import inspect
import json
import utils
from torch import nn
from copy import deepcopy
from pprint import pprint
import argparse
from functools import partial

from stable_baselines3 import SAC, HerReplayBuffer, DQN, PPO
from stable_baselines3.dqn import DQNwithICM
from stable_baselines3.sac import SACwithICM
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, MetaEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

#import imageio

from wandb.integration.sb3 import WandbCallback
import wandb

from datetime import datetime, timedelta
import time
import yappi

from goal_wrapper import GoalWrapper, FiveXGoalSelection, OrderedGoalSelection, GridNoveltySelection, FixedGoalSelection

from sparse_pendulum import SparsePendulumEnv
from pathological_mc import PathologicalMountainCarEnv

def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays_test = [np.asanyarray(arr) for arr in arrays]
    if not arrays_test:
        return np.array([[]])
    else:
        return np.stack(arrays, axis, out, dtype=None, casting="same_kind")

class MapGoalComponentsCallback(BaseCallback):
    def __init__(self, log_path, eval_points, dimension_names, eval_freq, goal_selector, verbose=0):
        super(MapGoalComponentsCallback, self).__init__(verbose)
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.eval_points = eval_points
        self.goal_selector = goal_selector
        self.dimension_names = dimension_names

    def _init_callback(self) -> None:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(os.path.join(self.log_path,"goal_component_map.txt"), "ab") as f:
                np.savetxt(f,
                        [],
                        header="step, " + str(self.dimension_names) \
                            + " experiment, expand, exclude, explain, exploit",
                        )



    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            components = self.goal_selector.select_goal_for_coverage(0,
                                                fixed_candidates=self.eval_points,
                                                map_not_choose=True,).T
            step = np.full((len(components), 1), self.n_calls)
            ret = np.concatenate((step, self.eval_points, components), 1)
            with open(os.path.join(self.log_path,"goal_component_map.txt"), "ab") as f:
                np.savetxt(f,ret)

        return True # never stop training

class ChangeParamCallback(BaseCallback):
    # TODO this class is extremely hacky, replace later
    def __init__(self,
                 total_steps,
                 object,
                 val_name_chain,
                 start_val,
                 end_val,
                 change_fraction=None,
                 verbose=0):
        super(ChangeParamCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.change_fraction = change_fraction # used if binary switch, gradual otherwise
        self.triggered = False # used if binary switch, gradual otherwise
        self.object = object
        self.val_name_chain = val_name_chain
        self.start_val = start_val
        self.end_val = end_val

    def set_val(self, prev_attr, remaining_name_chain, val):
        if len(remaining_name_chain) == 1:
            setattr(prev_attr, remaining_name_chain[0], val)
            if self.verbose >= 2:
                print(f"Set {remaining_name_chain[0]} to \
                    {getattr(prev_attr, remaining_name_chain[0], val)} \
                        in {prev_attr}")
            return
        attr = getattr(prev_attr, remaining_name_chain.pop(0))
        self.set_val(attr, remaining_name_chain, val)

    def _on_training_start(self):
        self.set_val(self.object, self.val_name_chain.copy(), self.start_val)

    def _on_rollout_end(self) -> bool:
        if self.change_fraction is not None and \
           not self.triggered and \
           self.num_timesteps > self.total_steps*self.change_fraction:
            self.triggered = True
            self.set_val(self.object, self.val_name_chain.copy(), self.end_val)
            return True # ends training early if False
        if self.change_fraction is None:
            val = self.start_val \
                + (self.num_timesteps/self.total_steps)*(self.end_val-self.start_val)
            self.set_val(self.object, self.val_name_chain.copy(), val)
            return True # ends training early if False

    def _on_step(self) -> bool:
        return super()._on_step()

def train(base_path: str = "./data/wrapper/",
          steps: int = 10000,
          experiments: int = 1,
          goal_weight: float = 0.5,
          goal_range: float = -1,
          eval_seed: int = None,
          train_seed: int = None,
          policy_seed: int = None,
          fixed_goal_fraction = 0.0,
          device = None,
          goal_selection_params: dict = None,
          after_goal_success: str = "term",
          range_as_goal: bool = False,
          env_params_override: dict = None,
          env_id = "PathologicalMountainCar-v1.1",
          baseline_override = None, # alternatives: "base-rl", "curious", "uniform-goal"
          verbose = 0,
          test_needed_experiments = False,
          algo_override = None,
          n_sampled_goal = 4, # Create 4 artificial transitions per real transition
          t2g_ratio = (1, "raw"), # train2gather ratio, first ratio, last "her" if mult with her_factor
          test_run = False,
          algo_kwargs =  None,
         ):

    if test_run:
        base_path = os.path.join(base_path,env_id+"|test_run")
    else:
        base_path = os.path.join(base_path,env_id)
    n_training_envs = 1
    n_eval_envs = 5

    if env_id == "PathologicalMountainCar-v1.1" or \
        env_id == "CliffWalking-v0":
        eval_freq = 5_000 #50_000 for patho MC and Cliffwalk, at least with goals
    elif env_id == "FrozenLake-v1":
        eval_freq = 10_000
    elif env_id == "MountainCar-v0":
        if baseline_override == "base-rl":
            eval_freq = 2_000 #50_000
        else:
            eval_freq = 20_000
    elif env_id == "Acrobot-v1":
        eval_freq = 2_000 #50_000
    else:
        raise "Error, set eval_freq for this env"

    pend_env_params = {"harder_start": [0.1]}
    pathmc_env_params = {"terminate": [True]}
    frozen_env_params = {"is_slippery": [True]}
    cliffwalker_env_params = {"max_episode_steps": [300]} # override to play nice with HER #{"is_slippery": [False]}

    if env_id == "SparsePendulumEnv-v1":
        env_params_default = pend_env_params
    elif env_id == "PathologicalMountainCar-v1.1":
        env_params_default = pathmc_env_params
    elif env_id == "FrozenLake-v1":
        env_params_default = frozen_env_params
    elif env_id == "CliffWalking-v0":
        env_params_default = cliffwalker_env_params
    elif env_id == "MountainCar-v0":
        env_params_default = {}
    elif env_id == "Acrobot-v1":
        env_params_default = {}
    else:
        print("env without env params?")
        exit()

    env_params = env_params_default | env_params_override

    # Collect variables to store in json before cluttered
    conf_params = locals()

    # TODO clean up this mess
    if baseline_override in [None, "grid-novelty"]:
        options = np.format_float_scientific(steps,trim="-",exp_digits=1) + "steps|" \
                + str(after_goal_success) \
                #+ str(goal_weight) + "goalrewardWeight_" \
                #+ str(fixed_goal_fraction) + "fixedGoalFraction_" \
                #+ str(goal_range) + "goal_range_" \

    if baseline_override in [None, "grid-novelty"]:
        options += utils.dict2string(goal_selection_params)
    elif baseline_override == "uniform-goal":
        options = np.format_float_scientific(steps,trim="-",exp_digits=1) + "steps|" \
                + str(after_goal_success) + "|" \
                + baseline_override + "-baseline"
                #+ str(goal_weight) + "goalrewardWeight_" \
                #+ str(goal_range) + "goal_range_" \

    if baseline_override in ["base-rl", "curious"]: # non-goal
        options = np.format_float_scientific(steps,trim="-",exp_digits=1) + "steps|" \
                + baseline_override + "-baseline"
    else:                                           # shared goal additions
        if range_as_goal:
            options += "|rangeGoal"
        else:
            options += str(goal_range) + "|goal_range"

    if fixed_goal_fraction != 0:
        options += "|" + str(fixed_goal_fraction) + "fixedGoalFraction"

    if n_sampled_goal != 4 and baseline_override not in ["base-rl"]:
        options += "|" + str(n_sampled_goal+1) + "her_factor"

    if t2g_ratio != (1, "raw"):
        if t2g_ratio[1] == "her":
            options += "|herX" + str(t2g_ratio[0]) + "t2g"
        else:
            options += "|" + str(t2g_ratio[0]) + "t2g"

    if algo_override is not None:
        options += "|" + str(algo_override.__name__)

    if t2g_ratio[1] == "her":
        t2g = t2g_ratio[0]*(n_sampled_goal+1)
    else:
        t2g = t2g_ratio[0]

    if algo_kwargs:
        # TODO cleaup and make only show up if not default (requires) knowing selected algo
        options += "|" + utils.dict2string(algo_kwargs)

    if env_params_override:
        options += "|" + utils.dict2string(env_params_override)

    # TODO hard coded options addition
    #options += "_256-256-256-256-256net"
    #options += "_finetune_at0.7_1.0of_time"#"_no_hindsight_then"
    #options += "_gradually_reduce_hindsight_from_0.8_to_0"
    #options += "_her_episode_gradually_reduce_hindsight_from_0.8_to_0"
    #options += "_batch_size512_train_freq512_lr_1e-3_4x256resblock-x4net_exploration_fraction0.5_more_novelty_focus"
    #options += "_batch_size512_train_freq512_lr_1e-3_3x256net"#_more_novelty_focus"
    #options += "_batch_size512_lr_1e-3_2x256net_0.1-0.005_epsilonexplore_decay_to_0.5_gamma0.999" # _sigmoid_output _0.1-0.005_epsilonexplore_decay_to_0.5" # _sigmoid_output _0_epsilonexplore" # _0.1-0.01_epsilonexplore #_goalpos[-1.70,-0.02],[0.6,0.02,]"

    if len(options) > 255:
        options = np.format_float_scientific(steps,trim="-",exp_digits=1) + "steps|" \
                + baseline_override + "-baseline"
        options_dict = algo_kwargs | env_params_override
        options += "|" + utils.to_short_string(options_dict)

    print(options)

    if after_goal_success == "term": # Add case for or restructure to handle exact match too
        terminate_at_goal_in_real = False # not needed if real term
    else:
        terminate_at_goal_in_real = True

    if baseline_override == "grid-novelty" and range_as_goal and not fixed_goal_fraction == 1:
        # TODO replace last and here later, temp hack
        # when using ranges as goals and goal selection methods that
        # select points, use this to turn those goals into ranges
        wrap_for_range = True
    else:
        wrap_for_range = False

    # check existing experiments
    exp_paths = utils.get_all_folders(os.path.join(base_path, options))

    # remove unfinished experiments from list so they can be re-run
    # TODO find a less hacky solution
    exp_paths_completed = []
    for exp in exp_paths:
        if os.path.isfile(os.path.join(exp, "completed.txt")):
            exp_paths_completed.append(exp)

    exps = [os.path.basename(exp_path) for exp_path in exp_paths_completed]

    # if test_needed_experiments we are just verifying number of needed experiments
    if test_needed_experiments:
        return exps

    # abort if we have enough experiments
    if experiments <= len(exps):
        return

    # find and set next experiment name
    exp_ind = 1
    while True:
        experiment = "exp" + str(exp_ind)
        if experiment not in exps:
            break
        exp_ind += 1

    # Create log dir
    log_dir = os.path.join(base_path, options, experiment, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    # save config file in folder to keep track of parameters used
    #signature = inspect.signature(FiveXGoalSelection)
    signature = inspect.signature(GridNoveltySelection)
    default_goal_selection_params = {k: v.default
                                     for k, v in signature.parameters.items()
                                     if v.default is not inspect.Parameter.empty}
    merged = default_goal_selection_params | goal_selection_params
    conf_params["goal_selection_params"] = merged
    conf_params["after_goal_success"] = after_goal_success
    if baseline_override is not None:
        # if making a baseline, some parameters are ignored, remove from conf
        # for clarity
        del conf_params["goal_selection_params"]
        del conf_params["fixed_goal_fraction"]
    if baseline_override in ["base-rl", "curious"]:
        del conf_params["goal_weight"]
    if algo_override is not None:
        conf_params["algo_override"] = conf_params["algo_override"].__name__
    with open(os.path.join(base_path, options, experiment, 'config.json'), 'w') as fp:
        json.dump(conf_params, fp, indent=4)

    # Initialize a training environment with default parameters
    #train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, vec_env_cls=SubprocVecEnv)
    train_env = gym.make(env_id, **env_params) #, render_mode='human')
    if train_seed is not None:
        train_env.reset(seed=train_seed)

    # wrap with goal conditioning and monitor wrappers
    # TODO replace all these checks with non-listed logic
    if baseline_override in [None, "uniform-goal", "grid-novelty"]:
        # only use when training with goal
        train_env_goal = GoalWrapper(train_env,
                                     goal_weight=goal_weight,
                                     goal_range=goal_range,
                                     after_goal_success=after_goal_success,
                                     range_as_goal=range_as_goal,
                                     wrap_for_range=wrap_for_range,
                                     )
                                     #intrinsic_curiosity_module=ICM(train_env,
                                     #                               device))
    else:
        train_env_goal = train_env # TODO handle that this name becomes missleading
    train_env = Monitor(train_env_goal, log_dir)
    #train_env = VecMonitor(train_env, log_dir)

    if env_id == "SparsePendulumEnv-v1":
        fixed_goal = lambda obs: np.array([1.0, 0.0, 0.0])
        few_goals = [fixed_goal,
                     lambda obs: np.array([1.0, 0.0, 3.0]),
                     lambda obs: np.array([1.0, 0.0, -3.0]),]
        many_goals = few_goals.copy()
        many_goals += [lambda obs: np.array([0.7071, 0.7071, 0.0]),
                       lambda obs: np.array([-0.7071, -0.7071, 0.0]),
                       lambda obs: np.array([-0.7071, 0.7071, 0.0]),
                       lambda obs: np.array([0.7071, -0.7071, 0.0]),]
        coord_names = ["x", "y", "ang. vel."]
        # algo = SAC
        algo = SACwithICM
        if baseline_override in ["base-rl", "curious"]: # TODO this looks wrong, I don't think i use "curious" yet
            algo = SAC
    elif env_id == "PathologicalMountainCar-v1.1":
        fixed_goal = lambda obs: np.array([-1.70, -0.02,])
        few_goals = [fixed_goal,
                     lambda obs: np.array([0.6, 0.02,])]
        many_goals = few_goals.copy()
        many_goals += [lambda obs: np.array([-0.5, 0.04,]),
                       lambda obs: np.array([-0.5, -0.04,]),
                       lambda obs: np.array([-0.7, 0.0,])]
        max_goals = [[-1.70, -0.02,],
                     [0.6, 0.02,],
                     [-0.5, 0.04,],
                     [-0.5, -0.04,],
                     [-0.7, 0.0,],
                     [0.0, 0.0,],
                     [-1.1, 0.0,],
                     [-1.1, 0.03,],
                     [-1.1, -0.05,],
                     [0.3, 0.0],
                     [0.3, 0.05],
                     [0.3, -0.02],
                     ]
        if range_as_goal:
            fixed_goal = lambda obs: np.array([-1.70, -0.07, -1.60, 0.0,])
            # max_goals = [[-1.70, -0.07, -1.60, 0.0,],
            #              [0.5, 0.0, 0.6, 0.07,],
            # ]
            max_goals = [[-1.70, -0.07, -1.60, 0.0,],
                         [0.5, 0.0, 0.6, 0.07,],
                         [ 1.00085936e-01, -6.02729847e-02,  3.86779122e-01,  4.44181123e-02],
                         [-3.56716424e-01, -6.18883563e-02,  8.63878641e-02, -8.34396244e-03],
                         [ 2.18684093e-01,  3.90903167e-02,  5.02551005e-01,  4.80780549e-02],
                         [ 4.03650701e-01, -2.63528430e-02,  4.33245642e-01,  4.98868548e-02],
                         [-1.67531003e+00, -3.84533985e-02, -7.09176660e-01,  6.33504931e-02],
                         [-1.68231821e+00, -6.89519766e-02, -1.50134969e+00,  6.59849776e-02],
                         [ 3.43190789e-01, -6.05484093e-02,  4.81386639e-01,  4.72225634e-02],
                         [-1.36810730e+00, -6.38140400e-02,  4.19232501e-01, -2.41154414e-02],
                         [ 4.85135019e-01,  5.27836259e-02,  5.69161911e-01,  6.84890449e-02],
                         [ 6.97672883e-02,  1.13778990e-02,  5.47868144e-01,  4.15376470e-02],
                         [-1.26223329e+00,  6.89371377e-02,  3.42234741e-01,  6.99203823e-02],
                         [-2.32861340e-01, -6.84580506e-02,  4.57449734e-01,  3.24315328e-02],
                         [-1.51552431e+00,  1.84240937e-02, -1.95313994e-02,  2.09420945e-02],
                         [-1.62400395e+00,  3.14769447e-02, -7.20394119e-01,  6.43267009e-02],
                         [-1.37953661e+00,  1.36864912e-02,  5.97108342e-01,  2.75630166e-02],
                         [-3.51205835e-01,  6.73303157e-02,  5.14140035e-01,  6.81073994e-02],
                         [-1.61280006e+00, -6.01168698e-02,  3.67342031e-01, -1.87129658e-02],
                         [ 4.06076461e-01, -6.17086060e-02,  4.07544946e-01, -2.45365343e-02],
                         [-1.21682617e+00,  5.18246852e-02,  1.12338400e-01,  5.30956992e-02],
                         [-1.49977030e+00,  5.86865321e-02,  4.71921313e-01,  6.67302235e-02],
                         [ 3.07349324e-01, -6.35200740e-02,  3.79228258e-01,  2.55321586e-02],
                         [-4.29561395e-01,  2.63381153e-02,  2.90427584e-02,  4.70002601e-02],
                         [-1.55965815e+00, -4.15685597e-02, -7.91227965e-01, -9.37543996e-03],
                         [-6.11123538e-01,  1.01132356e-02,  1.75906278e-01,  3.32736568e-02],
                         [-7.85424886e-01, -1.35230819e-03, -2.13653274e-01, -5.12745173e-04],]
        coord_names = ["xpos", "velocity"]
        # algo = DQN
        algo = DQNwithICM # TODO we are not using ICM now, don't use it in any? or start using again?
        if baseline_override in ["base-rl", "curious", "uniform-goal", "grid-novelty"]: # TODO this looks wrong, I don't think i use "curious" yet
            algo = DQN
    elif env_id == "FrozenLake-v1":
        # TODO make and shift when running with 8x8
        fixed_goal = lambda obs: np.array([15])
        few_goals = [fixed_goal,
                     lambda obs: np.array([4])]
        many_goals = few_goals.copy()
        many_goals += [lambda obs: np.array([12]),
                       lambda obs: np.array([7]),
                       lambda obs: np.array([8])]
        max_goals = [i for i in range(16)]
        coord_names = ["index"]
        # algo = DQN
        algo = DQNwithICM
        if baseline_override in ["uniform-goal", "grid-novelty", "base-rl", "curious"]: # TODO this looks wrong, I don't think i use "curious" yet
            algo = DQN
    elif env_id == "CliffWalking-v0":
        # TODO make and shift when running with 8x8
        fixed_goal = lambda obs: np.array([47])
        few_goals = [fixed_goal,
                     lambda obs: np.array([4])]
        many_goals = few_goals.copy()
        many_goals += [lambda obs: np.array([12]),
                       lambda obs: np.array([7]),
                       lambda obs: np.array([8])]
        max_goals = [i for i in range(48)] # TODO grab from env obs space instead
        coord_names = ["index"]
        # algo = DQN
        algo = DQNwithICM
        if baseline_override in ["uniform-goal", "grid-novelty", "base-rl", "curious"]: # TODO this looks wrong, I don't think i use "curious" yet
            algo = DQN
    elif env_id == "MountainCar-v0":
        if range_as_goal:
            fixed_goal = lambda obs: np.array([0.5, 0.0, 0.6, 0.07,])
            max_goals = [
                [-1.70, -0.07, -1.60, 0.0,],
                [0.5, 0.0, 0.6, 0.07,],
                [ 1.00085936e-01, -6.02729847e-02,  3.86779122e-01,  4.44181123e-02],
                [-3.56716424e-01, -6.18883563e-02,  8.63878641e-02, -8.34396244e-03],
                [ 2.18684093e-01,  3.90903167e-02,  5.02551005e-01,  4.80780549e-02],
                [ 4.03650701e-01, -2.63528430e-02,  4.33245642e-01,  4.98868548e-02],
                [-1.67531003e+00, -3.84533985e-02, -7.09176660e-01,  6.33504931e-02],
                [-1.68231821e+00, -6.89519766e-02, -1.50134969e+00,  6.59849776e-02],
                [ 3.43190789e-01, -6.05484093e-02,  4.81386639e-01,  4.72225634e-02],
                [-1.36810730e+00, -6.38140400e-02,  4.19232501e-01, -2.41154414e-02],
                [ 4.85135019e-01,  5.27836259e-02,  5.69161911e-01,  6.84890449e-02],
                [ 6.97672883e-02,  1.13778990e-02,  5.47868144e-01,  4.15376470e-02],
                [-1.26223329e+00,  6.89371377e-02,  3.42234741e-01,  6.99203823e-02],
                [-2.32861340e-01, -6.84580506e-02,  4.57449734e-01,  3.24315328e-02],
                [-1.51552431e+00,  1.84240937e-02, -1.95313994e-02,  2.09420945e-02],
                [-1.62400395e+00,  3.14769447e-02, -7.20394119e-01,  6.43267009e-02],
                [-1.37953661e+00,  1.36864912e-02,  5.97108342e-01,  2.75630166e-02],
                [-3.51205835e-01,  6.73303157e-02,  5.14140035e-01,  6.81073994e-02],
                [-1.61280006e+00, -6.01168698e-02,  3.67342031e-01, -1.87129658e-02],
                [ 4.06076461e-01, -6.17086060e-02,  4.07544946e-01, -2.45365343e-02],
                [-1.21682617e+00,  5.18246852e-02,  1.12338400e-01,  5.30956992e-02],
                [-1.49977030e+00,  5.86865321e-02,  4.71921313e-01,  6.67302235e-02],
                [ 3.07349324e-01, -6.35200740e-02,  3.79228258e-01,  2.55321586e-02],
                [-4.29561395e-01,  2.63381153e-02,  2.90427584e-02,  4.70002601e-02],
                [-1.55965815e+00, -4.15685597e-02, -7.91227965e-01, -9.37543996e-03],
                [-6.11123538e-01,  1.01132356e-02,  1.75906278e-01,  3.32736568e-02],
                [-7.85424886e-01, -1.35230819e-03, -2.13653274e-01, -5.12745173e-04],]
        else:
            fixed_goal = lambda obs: np.array([0.5, 0.0, ])
        coord_names = ["xpos", "velocity"]
        algo = DQN
    elif env_id == "Acrobot-v1":
        if range_as_goal:
            # TODO
            print("No range goals for Acrobot-v1 yet, exiting")
            exit()
        else:
            fixed_goal = lambda obs: np.array([-1.0, 0.0, -1.0, 0.0, 0.0, 0.0])
        coord_names = ["cos_theta1", "sin_theta1","cos_theta1", "sin_theta1","theta1dot","theta2dot"]
        algo = DQN

    if algo_override is not None:
        algo = algo_override

    def setup_evalcallbacks_from_goal_list(goals, freq=eval_freq, n_eval_episodes=1):

        eval_callback_list = []
        for i, goal in enumerate(goals):
            eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs_" + str(goal))
            os.makedirs(eval_log_dir, exist_ok=True)

            eval_env = gym.make(env_id, **env_params)
            if eval_seed is not None:
                eval_env.reset(seed=eval_seed)
            goal_selector = FixedGoalSelection(goal)
            goal_selection = goal_selector.select_goal
            goal_weight=1
            eval_goal_range=0.1

            eval_env_goal = GoalWrapper(eval_env,
                                goal_weight=goal_weight,
                                goal_range=eval_goal_range,
                                goal_selection_strategies=goal_selection,
                                after_goal_success="term",
                                range_as_goal=range_as_goal,
                                )
                                # TODO this will capture when goals are reached according to own metric
                                # but will miss when goals are "failed" because of external termination
                                # even when that termination is for the goal we aim to care for, so
                                # goal_range should not be too small here, so that does not happen
            eval_env = Monitor(eval_env_goal, eval_log_dir)

            eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=eval_log_dir,
                                    log_path=eval_log_dir,
                                    log_name=str(i), # TODO index or shorted goal as name?
                                    eval_freq=max(freq // n_training_envs, 1),
                                    n_eval_episodes=n_eval_episodes,
                                    deterministic=True,
                                    render=False,
                                    verbose=verbose,
                                    seed=eval_seed)
            eval_callback_list.append(eval_callback)
        return eval_callback_list


    def setup_eval(eval_type):
        # Create log dir where evaluation results will be saved
        if eval_type == "fixed":
            eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
        else:
            eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs" + eval_type)
        os.makedirs(eval_log_dir, exist_ok=True)

        # Separate evaluation env, with different parameters passed via env_kwargs
        # Eval environments can be vectorized to speed up evaluation.
        #eval_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
        #eval_env = GoalPendulumEnv(render_mode="human",
        #                           fixed_goal=np.array([1.0, 0.0, 0.0]))
        eval_env = gym.make(env_id, **env_params)#,render_mode="human")
        if eval_seed is not None:
            eval_env.reset(seed=eval_seed)

        if eval_type == "few":
            goal_selector = OrderedGoalSelection(few_goals)
            goal_selection = goal_selector.select_goal
            goal_weight=1
            n_eval_episodes=goal_selector.len
            after_goal_success="term"
            eval_goal_range=0.1
        if eval_type == "many":
            goal_selector = OrderedGoalSelection(many_goals)
            goal_selection = goal_selector.select_goal
            goal_weight=1
            n_eval_episodes=goal_selector.len
            after_goal_success="term"
            eval_goal_range=0.1
        if eval_type == "fixed":
            goal_selection = fixed_goal
            goal_weight=0
            n_eval_episodes=5
            after_goal_success="reselect" # to make sure we don't terminate early when going for external reward
            eval_goal_range=0.0001


        # for eval we want to always evaluate with the goal at the top
        # if weight should be 0 (using real reward, or 1 using goal reward, or a mixture like
        # in training) can be discussed
        # TODO replace all these checks with non-listed logic
        if baseline_override in [None, "uniform-goal", "grid-novelty"]:
            # only evaluate with goal when training with goal
            eval_env_goal = GoalWrapper(eval_env,
                                goal_weight=goal_weight,
                                goal_range=eval_goal_range,
                                goal_selection_strategies=goal_selection,
                                after_goal_success=after_goal_success,
                                range_as_goal=range_as_goal,
                                )
                                # for goal_weight 1 cases goal_range should probably not be too small
                                # for goal_weight 0 cases goal_range should probably be small if we
                                # just look for external reward, ideally tiny
        else:
            eval_env_goal = eval_env # TODO handle that this name becomes missleading
        eval_env = Monitor(eval_env_goal, eval_log_dir)

        if eval_type == "fixed":
            name = "hard"
        else:
            name = ""

        # call dedicated utility function: meta_eval_reward_quick_and_no_v_explode
        if True:
            meta_eval_n = "meta_eval/meta_value"
            func = utils.meta_eval_reward_quick_and_no_v_explode
            meta_callback = MetaEvalCallback(eval_func=func,
                                            vars_to_eval=["last_mean_reward",
                                                          "last_mean_initial_values",
                                                          "n_calls",
                                                          "num_timesteps",
                                                          ],
                                            log_name=meta_eval_n,
                                            window=steps,
                                            log_val=True,
                                            #log_best="+",
                                            #step_of_bool="-"
                                            log_on_end_only=True,
                                            )

        # meta_eval steps to get window average of reward
        if False:
            window = 10
            windowed_reward_n = f"avg_reward_for_{window}_evals"
            func = np.mean
            meta_callback = MetaEvalCallback(eval_func=func,
                                            vars_to_eval="last_mean_reward",
                                            log_name=windowed_reward_n,
                                            window=window,
                                            log_val=True,
                                            log_best="+",
                                            #step_of_bool="-"
                                            )

        # meta_eval steps to get window average over thresh
        if False:
            window = 5
            thresh = -130
            if window is not None and window > 0:
                windowed_reward_n = f"avg_reward_over_{thresh}_for_{window}_evals"
            else:
                windowed_reward_n = f"over_reward_{thresh}_for_{window}_evals"
            func = partial(utils.check_threshold, thresh=thresh)
            meta_callback = MetaEvalCallback(eval_func=func,
                                            vars_to_eval="last_mean_reward",
                                            log_name=windowed_reward_n,
                                            window=window,
                                            log_val=False,
                                            #log_best="+",
                                            #log_step="-",
                                            step_of_bool="-"
                                            )

        # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
        # When using multiple training environments, agent will be evaluated every
        # eval_freq calls to train_env.step(), thus it will be evaluated every
        # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
        # TODO set identical env initiations in eval
        eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=eval_log_dir,
                                    log_path=eval_log_dir,
                                    log_name=name, # TODO shorten or use index?
                                    eval_freq=max(eval_freq // n_training_envs, 1),
                                    n_eval_episodes=n_eval_episodes,
                                    deterministic=True,
                                    render=False,
                                    verbose=verbose,
                                    seed=eval_seed,
                                    callback_after_eval=meta_callback,
                                    )

        return eval_callback

    eval_callback = setup_eval("fixed")
    # TODO replace all these checks with non-listed logic
    if baseline_override != "base-rl":
        clist = setup_evalcallbacks_from_goal_list(max_goals, n_eval_episodes=1)
        clist.append(eval_callback)
        eval_callbacks = clist
    else:
        eval_callbacks = [eval_callback]

    eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    # TODO replace all these checks with non-listed logic
    if baseline_override in [None, "uniform-goal", "grid-novelty"]:
        # only use HER buffer when training with goal
        policy = "MultiInputPolicy"
        algo_kwargs_add = {"replay_buffer_class": HerReplayBuffer,
                       "replay_buffer_kwargs": dict(
                       n_sampled_goal=n_sampled_goal,
                       goal_selection_strategy="future",
                       copy_info_dict=goal_weight!=1.0, # TODO Turn off if not needed
                       terminate_at_goal=True, # TODO make dependant on goal eval strat
                       terminate_at_goal_in_real=terminate_at_goal_in_real,), # TODO make dependant on goal eval strat
                       }
        # if env_id == "CliffWalking-v0":
        #     algo_kwargs["replay_buffer_kwargs"]["handle_timeout_termination"] = True
    else:
        policy = "MlpPolicy"
        algo_kwargs_add = {}

    # setup algo_kwargs_default
    #policy_kwargs=dict(net_arch=[256, 256, 256],)
    #policy_kwargs=dict(net_arch=[128, 128, 128],)
    #policy_kwargs=dict(net_arch=[256, 256],)
    policy_kwarg_defaults=dict(net_arch=[64, 64],)
    #policy_kwargs=dict(net_arch=[256, 256, 256, 256, 256, 256, 256, 256, 256],)
    #policy_kwargs=dict(net_arch={"res-block": [256, 256, 256, 256], "num_blocks": 4},)
    if baseline_override != "base-rl":
        policy_kwarg_defaults["out_act_fn"] = nn.Sigmoid()

    # TODO make default from algo __init__ defaults first, then do any overrides
    # with my defaults here, and that will be my default conf

    batch_size = 128 # 512
    learning_starts = 500
    if "max_episode_steps" in env_params:
        learning_starts = max(learning_starts, env_params["max_episode_steps"])

    algo_kwargs_default = dict(
        learning_starts=500, #300
        verbose=verbose,
        buffer_size = int(1e6),
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=batch_size,
        train_freq=1, #max(int(batch_size/t2g), 1),
        target_update_interval=100, # TODO this was prob the error! try arround here
        #gradient_steps=-1,
        policy_kwargs=policy_kwarg_defaults, # TODO put overrides in finial used
        seed=policy_seed,
        device=device,
        tensorboard_log=log_dir,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.5,
        double_dqn=False,
    )

    temp = utils.deepcopy(algo_kwargs_default)
    # Add HER replay buffer stuff to default if GC
    utils.deepmerge(algo_kwargs_default, algo_kwargs_add)

    algo_kwargs_merged = utils.deepmerge({},
                                         algo_kwargs_default,
                                         algo_kwargs)

    pprint(algo_kwargs_merged)
    diff = utils.DeepDiff(algo_kwargs_default,
                          algo_kwargs_merged)
    # diff = utils.DeepDiff(temp,
    #                       algo_kwargs_merged,)
    delta = utils.Delta(diff, force=True)
    diff_dict = {} + delta
    diff_dict = utils.replace_type_in_dict_from(diff_dict, algo_kwargs_merged)
    group_name = utils.to_short_string(diff_dict, temp)

    # TODO if group name over 128 characters, deal with it better
    if len(group_name) > 128:
        group_name = group_name[:128]

    print(f"Group name: {group_name}")

    model = algo(policy,
                train_env,
                **algo_kwargs_merged,
    )

    #model.replay_buffer.her_ratio = 0
    # eval_callbacks.append(ChangeParamCallback(total_steps=steps,
    #                                          object=model,
    #                                          val_name_chain=["replay_buffer", "her_ratio"],
    #                                          start_val=0.8,
    #                                          end_val=0,
    #                                          change_fraction=None))

    if baseline_override is None:
        # TODO a smart goal selection needs access to buffer of seen states and of targeted goals
        # policy algorithm can be mostly separate, but we should link up with its buffer when it exists
        # and create one otherwise. We should therefore set goal strategy here after model is setup I guess
        # removes some nice separation, but is maybe ok for now?
        # Also, when we replace the set of all seen states with a downsamples proxy, this might be more clear
        # train_env_goal.link_buffer(model.replay_buffer)
        if goal_selection_params is not None: # assumes that all keys are params to func
            goal_selection = FiveXGoalSelection(train_env_goal,
                                                model.replay_buffer,
                                                train_env_goal.initial_targeted_goals,
                                                train_steps=steps,
                                                verbose = verbose > 0,
                                                **goal_selection_params,)
        else:
            goal_selection = FiveXGoalSelection(train_env_goal,
                                                model.replay_buffer,
                                                train_env_goal.targeted_goals,
                                                train_steps=steps,)

        value_map_grid = goal_selection.grid_of_points(10)
        mapping_callback = MapGoalComponentsCallback(log_path=eval_log_dir,
                                                    eval_points=value_map_grid,
                                                    eval_freq=max(500 // n_training_envs, 1),
                                                    goal_selector=goal_selection,
                                                    dimension_names="x, y, theta,"
                                                    )

        #callback = CallbackList([mapping_callback, eval_callback])
        #callback = CallbackList([eval_callback])
        callbacks = eval_callbacks

        #goal_selection_strategies = [train_env_goal.sample_obs_goal, fixed_goal]
        goal_selection_strategies = [goal_selection.select_goal_for_coverage, fixed_goal]
        goal_sel_strat_weight = [1-fixed_goal_fraction, fixed_goal_fraction]
        train_env_goal.set_goal_strategies(goal_selection_strategies, goal_sel_strat_weight)
        train_env_goal.print_setup()
    elif baseline_override == "uniform-goal":
        callbacks = eval_callbacks
        train_env_goal.set_goal_strategies([train_env_goal.sample_obs_goal])
        train_env_goal.print_setup()
    elif baseline_override == "grid-novelty":
        callbacks = eval_callbacks
        goal_selection = GridNoveltySelection(train_env_goal,
                                              train_steps=steps,
                                              **goal_selection_params,
                                              )
        goal_selection_strategies = [goal_selection.select_goal, fixed_goal]
        goal_sel_strat_weight = [1-fixed_goal_fraction, fixed_goal_fraction]
        train_env_goal.set_goal_strategies(goal_selection_strategies,
                                           goal_sel_strat_weight,
                                           goal_selector_obj=goal_selection)
        train_env_goal.print_setup()
    else:
        callbacks = eval_callbacks

    # TODO update this when updating code above here, and input to this method
    # alternatively, move all these checks and name stuff to external function,
    # so this can be done there based on experiment_list in main method prob best
    wandb_config = dict(
        steps = steps,
        goal_weight = goal_weight,
        goal_range = goal_range,
        eval_seed = eval_seed,
        train_seed = train_seed,
        policy_seed = policy_seed,
        fixed_goal_fraction = fixed_goal_fraction,
        device = device,
        goal_selection_params = goal_selection_params,
        after_goal_success = after_goal_success,
        range_as_goal = range_as_goal,
        env_params = env_params,
        env_id = env_id,
        baseline_override = baseline_override,
        verbose = verbose,
        test_needed_experiments = test_needed_experiments,
        algo_override = algo_override,
        n_sampled_goal = n_sampled_goal,
        t2g_ratio = t2g_ratio,
        algo_kwargs = algo_kwargs_merged,
        eval_freq = eval_freq,
        terminate_at_goal_in_real = terminate_at_goal_in_real,
        wrap_for_range = wrap_for_range,
    )

    # TODO determine if this needs to be after any of the things below like the
    # algo_kwarg stuff, or if the algorithm kwargs are captured anyway
    # setup wandb after confirming experiments, this is where we know we are running
    proj_name = env_id
    # if env params changed, add to proj name, as it different task then
    if env_params_override is not None:
        proj_name += "|" + utils.dict2string(env_params_override)

    if test_run:
        proj_name += "|test_run"

    run = wandb.init( # TODO determine how to do this best
        project=proj_name,
        dir=os.path.join(base_path, options, experiment, "wandb_logs"),
        # name=??, # should be short, how to make shortest here?
        notes=options, # put options as note instead
        tags=[baseline_override, after_goal_success],
        config=wandb_config,
        group=group_name,
        save_code=True,  # optional
        #sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        reinit="finish_previous",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=eval_freq/10, # TODO make separate variable for this
        model_save_freq=eval_freq,
        model_save_path=os.path.join(base_path, options, experiment, 'model'),
    )

    callbacks.append(wandb_callback)

    # replace logger to add wandb stuff to it as well
    format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")
    format_strings.append(run) # TODO this breaks type hint for configure
    new_logger = configure(log_dir, format_strings)
    model.set_logger(new_logger)

    print(model.policy)

    start_time = time.time()
    model.learn(steps, callback=CallbackList(callbacks), progress_bar=True)
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    #model.learn(steps, progress_bar=True)
    model_path = os.path.join(base_path, options, experiment, 'final_model')
    model.save(model_path) # TODO this is probably no longer needed, but check first

    if baseline_override in ["grid-novelty"]:
        np.savetxt(os.path.join(base_path, options, experiment,"n_shape"),
                   goal_selection.shape)
        if hasattr(goal_selection, "cell_size"):
            np.savetxt(os.path.join(base_path, options, experiment,"n_cell_size"),
                    goal_selection.cell_size)
        np.savetxt(os.path.join(base_path, options, experiment,"n_visits"),
                   goal_selection.visits)
        np.savetxt(os.path.join(base_path, options, experiment,"n_last_visit"),
                   goal_selection.last_visit)
        np.savetxt(os.path.join(base_path, options, experiment,"n_targeted"),
                   goal_selection.targeted)
        np.savetxt(os.path.join(base_path, options, experiment,"n_succeded"),
                   goal_selection.succeded)
        np.savetxt(os.path.join(base_path, options, experiment,"n_succeded_obs"),
                   goal_selection.succeded_obs)
        np.savetxt(os.path.join(base_path, options, experiment,"n_latest_targeted"),
                   goal_selection.latest_targeted)
        np.savetxt(os.path.join(base_path, options, experiment,"n_latest_succeded"),
                   goal_selection.latest_succeded)

    # mark experiment complete
    with open(os.path.join(base_path, options, experiment, "completed.txt"), 'w') as fp:
        # hacky extra info to be able to read monitor data instead of evaluation.npz
        # as it has the wrong metric (raw reward instead of both that and if applicable
        # success rate)
        if env_id == "PathologicalMountainCar-v1.1":
            fp.write('eval_logs 5\n')
            fp.write('eval_logsfew 2\n')
            fp.write('eval_logsmany 5\n')
        elif env_id == "SparsePendulumEnv-v1":
            fp.write('eval_logs 5\n')
            fp.write('eval_logsfew 2\n')
            fp.write('eval_logsmany 5\n')
        elif env_id == "FrozenLake-v1":
            fp.write('eval_logs 5\n')
        elif env_id == "CliffWalking-v0":
            fp.write('eval_logs 5\n')
        elif env_id == "MountainCar-v0":
            fp.write('eval_logs 5\n')
            fp.write('eval_logsfew 2\n')
            fp.write('eval_logsmany 5\n')
        elif env_id == "Acrobot-v1":
            fp.write('eval_logs 5\n')
            fp.write('eval_logsfew 2\n')
            fp.write('eval_logsmany 5\n')

    run.finish()

    if baseline_override in [None, "uniform-goal", "grid-novelty"]:
        targeted_goals = stack(train_env_goal.targeted_goals)
        initial_targeted_goals = stack(train_env_goal.initial_targeted_goals)
        successful_goals = stack(train_env_goal.successful_goals)
        success_obs = stack(train_env_goal.success_obs)
        successful_goal_index = stack(train_env_goal.successful_goal_index)
        local_targeted_goals = stack(train_env_goal.local_targeted_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"goals"), targeted_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"initial_targeted_goals"), initial_targeted_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"successful_goal_spread"), successful_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"successful_goal_obs_spread"), success_obs)
        np.savetxt(os.path.join(base_path, options, experiment,"successful_goal_index"), successful_goal_index)
        np.savetxt(os.path.join(base_path, options, experiment,"reselect_goal_spread"), local_targeted_goals)

        if len(targeted_goals.shape) > 1: # requires goals tracked in more than one dim to plot this way
            utils.plot_targeted_goals(targeted_goals,
                                    coord_names,
                                    os.path.join(base_path, options, experiment),
                                    figname="goal_spread")
            utils.plot_targeted_goals(initial_targeted_goals,
                                    coord_names,
                                    os.path.join(base_path, options, experiment),
                                    figname="ep_start_goal_spread")
            if len(train_env_goal.local_targeted_goals) > 0:
                utils.plot_targeted_goals(local_targeted_goals,
                                        coord_names,
                                        os.path.join(base_path, options, experiment),
                                        figname="reselect_goal_spread")
            utils.plot_targeted_goals(successful_goals,
                                    coord_names,
                                    os.path.join(base_path, options, experiment),
                                    figname="successful_goal_spread")
            utils.plot_targeted_goals(success_obs,
                                    coord_names,
                                    os.path.join(base_path, options, experiment),
                                    figname="successful_goal_obs_spread")

#model.load(model_path, eval_env)

#model.set_env(eval_env)
#obs, info = eval_env.reset()

# Enjoy trained agent
#print(obs)
#for i in range(200):
#    action, _states = model.predict(obs)
#    obs, rewards, terms, trunks, infos = eval_env.step(action)
#    eval_env.render()

# images = []
# img = model.env.render(mode='rgb_array')
# for i in range(350):
#     images.append(img)
#     action, _ = model.predict(obs)
#     obs, _, _ ,_ = model.env.step([action])
#     img = model.env.render(mode='rgb_array')

#imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

# num = 1000

# distances = []
# rewards = []
# for i in range(num):
#     obs1, _ = train_env.reset()
#     obs2, _ = train_env.reset()
#     distance = np.linalg.norm((obs1["desired_goal"] - obs2["desired_goal"])/train_env.obs_norm, axis=-1)
#     reward = np.exp(-distance)
#     distances.append(distance)
#     rewards.append(reward)

# for res in [distances, rewards]:
#     print(f"Mean: {np.mean(res)}")
#     print(f"Std: {np.std(res)}")
#     print(f"Median: {np.median(res)}")
#     print(f"Min: {np.min(res)}")
#     print(f"Max: {np.max(res)}")

#     print(np.sort(res))

def named_permutations(params_to_permute: dict):
    list = []
    for combs in product (*params_to_permute.values()):
        conf = {ele: cnt for ele, cnt in zip(params_to_permute, combs)}
        list.append(conf)
    return list

def confirm():
    """
    Ask user to enter Y or N (case-insensitive).

    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Execute these experiments [Y/N]? ").lower()
    return answer == "y"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_run', action='store_true')
    args = parser.parse_args()
    profile = False
    if profile:
        yappi.start()

    # temp note of params that worked for Acrobot
    # policy_kwargs_to_permute=dict(
    #     net_arch=[[80],] # [80], [32, 32],[128, 128], [64, 64], [64, 64, 64],
    # )

    # algo_kwargs_to_permute = dict(
    #     policy_kwargs=named_permutations(policy_kwargs_to_permute),
    #     learning_rate=[1e-3,], #1e-3
    #     batch_size=[200], # 256
    #     buffer_size=[5_000,],
    #     target_update_interval=[100], # 10,100,200,500,1000
    #     gamma=[0.99,], # 0.95, 0.99,0.999
    #     exploration_fraction=[0.25,], # 0.5,0.2,0.1
    #     exploration_initial_eps=[1.0], # 1,0.1,0.01,
    #     exploration_final_eps=[0.01],
    #     train_freq=[1,], # 1,3,10,30
    #     double_dqn=[False],
    #     learning_starts=[500], # 500,1000,3000,10_000
    #     #tau=[0.2,0.3], # 1.0, 0.5, 0.1
    # )

    goal_conf_to_permute = {"grid_size": [100],
                            "fraction_random": [0.01], # 0.01
                            #"target_success_rate": [0.75], # turn on if intermediate instead of novelty # 0.75
                            "dist_decay": [2],
                            }

    env_id = "MountainCar-v0" # "Acrobot-v1" # "MountainCar-v0" "CliffWalking-v0" "PathologicalMountainCar-v1.1" # "FrozenLake-v1" "PathologicalMountainCar-v1.1" "SparsePendulumEnv-v1"
    # TODO get updated gym envs with cliffwalking v1

    env_params_override = {"max_episode_steps": [1000]}

    policy_kwargs_to_permute=dict(
        net_arch=[[128, 128],] # [80], [32, 32],[128, 128], [64, 64], [64, 64, 64],
    )

    algo_kwargs_to_permute = dict(
        policy_kwargs=named_permutations(policy_kwargs_to_permute),
        learning_rate=[2e-3,], #1e-3
        batch_size=[256], # 256
        buffer_size=[100_000,], # also try 400_000
        target_update_interval=[4500], # 10,100,200,500,1000
        gamma=[0.99,], # 0.95, 0.99,0.999
        exploration_fraction=[0.85,], # 0.5,0.2,0.1
        exploration_initial_eps=[0.35], # 1,0.1,0.01,
        exploration_final_eps=[0.001],
        train_freq=[15,], # 1,3,10,30
        double_dqn=[True],
        learning_starts=[17_000], # 500,1000,3000,10_000
        #tau=[0.2,0.3], # 1.0, 0.5, 0.1
    )

    params_to_permute = {"experiments": [25],
                         "env_id": [env_id],
                         "fixed_goal_fraction": [0.0],
                         "device": ["cpu"],
                         "steps": [1_000_000], # 10_000_000 pmc default
                         "goal_weight": [1.0],
                         "goal_range": [0.0001],
                         "goal_selection_params": named_permutations(goal_conf_to_permute),
                         "env_params_override": named_permutations(env_params_override),
                         "algo_kwargs": named_permutations(algo_kwargs_to_permute),
                         "after_goal_success": ["reselect"], # "exact_goal_match_reward" "term", "reselect", "local-reselect" # only applies to train, eval terms
                         # TODO reward_func is replaced with a pure term or reselect param
                         # but we also need one for reward eval and handle it's relation to
                         # goal selection methods
                         "baseline_override": ["base-rl"], # ["base-rl", "uniform-goal", "grid-novelty"] [None]  # should be if not doing baseline None
                         "range_as_goal": [True], # only works with uniform goal selection for now
                         #"algo_override": [PPO],
                         "n_sampled_goal": [4],
                         "t2g_ratio": [(1, "raw")], #first part ratio, last "raw" if raw or "her" on top of her ratio
                         }

    base_path = "./output/wrapper/"

    experiment_list = named_permutations(params_to_permute)

    wandb.login()
    # wandb.tensorboard.patch(root_logdir=os.path.join(base_path))

    if args.test_run:
        print("This is a test run")

    # run test runs first to confirm needed number of experiments
    full_experiment_list = []
    total_planned = 0
    total_found = 0
    for conf in experiment_list:
        completed_exps = train(base_path = base_path,
                               verbose = 0,
                               test_needed_experiments = True,
                               test_run = args.test_run,
                               **conf)
        total_planned += conf["experiments"]
        total_found += min(len(completed_exps), conf["experiments"])
        if len(completed_exps) < conf["experiments"]:
            full_experiment_list += [conf]*(conf["experiments"]-len(completed_exps))

    print(f"Out of {total_planned} planned experiments, {total_found} were already completed")
    print(f"{len(full_experiment_list)} experiments queued up to be run")
    if not confirm():
        exit()
    total_time = 0 # in seconds
    for i, conf in enumerate(full_experiment_list):
        print("Training with configuration: " + str(conf))
        start_time = time.monotonic()

        train(base_path = base_path,
              verbose = 0,
              eval_seed = 2, # seeds 2,3 on the easeir side in path MC, seed 0 at the bottom, 4 on hard side
              test_run = args.test_run,
              **conf)

        time_used = time.monotonic() - start_time
        total_time += time_used
        #part_time = time.strftime('%H:%M:%S', time.gmtime(time_used))
        part_time = timedelta(seconds=time_used)
        print(f"--- latest experiment took {part_time} ---")
        total_estimate = total_time/(i+1) * len(full_experiment_list)
        #completed_time = time.strftime('%H:%M:%S', time.gmtime(total_time))
        #total_time_estimate = time.strftime('%H:%M:%S', time.gmtime(total_estimate))
        completed_time = timedelta(seconds=total_time)
        total_time_estimate = timedelta(seconds=total_estimate)
        print(f"--- {i+1}/{len(full_experiment_list)} experiments have been " \
            + f"completed in {completed_time}/{total_time_estimate} (total time estimated) ---")

    if profile:
        yappi.stop()
        yappi.get_func_stats().save("./profile1milStep_10kBuffer.pstats",type="pstat")

    # for conf in product(fixed_goal_fractions, experiments):
    #     print("Training with configuration: " + str(conf))
    #     train(fixed_goal_fraction = conf[0],
    #             experiment=conf[1],
    #             steps=20000,
    #             goal_weight=1.0,
    #             device="cuda"
    #             )