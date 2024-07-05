import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import inspect
import json
import utils

from stable_baselines3 import SAC, HerReplayBuffer, DQN
from stable_baselines3.dqn import DQNwithICM
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

#import imageio

import time
import yappi

from goal_wrapper import GoalWrapper, FiveXGoalSelection

from sparse_pendulum import SparsePendulumEnv
from pathological_mc import PathologicalMountainCarEnv

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
          env_params: dict = None,
          env_id = "PathologicalMountainCar-v1.1",
          buffer_size = int(1e6),
          baseline_override = None, # alternatives: "base-rl", "curious", "uniform-goal"
          verbose = 0,
          test_needed_experiments = False,
         ):

    base_path = os.path.join(base_path,env_id)
    n_training_envs = 1
    n_eval_envs = 5

    # Create 4 artificial transitions per real transition
    n_sampled_goal = 4

    # Collect variables to store in json before cluttered
    conf_params = locals()

    if baseline_override is None:
        options = str(steps) + "steps_" \
                + str(goal_weight) + "goalrewardWeight_" \
                + str(fixed_goal_fraction) + "fixedGoalFraction_" \
                + str(goal_range) + "goal_range"

        for key, val in goal_selection_params.items():
            if type(val) is list:
                options += "_[" + ",".join(str(v) for v in val) + "]" + key
            elif val == True:
                options += "_" + key
            else:
                options += "_" + str(val) + key
    elif baseline_override == "uniform-goal":
        options = str(steps) + "steps_" \
                + str(goal_weight) + "goalrewardWeight_" \
                + str(goal_range) + "goal_range_" \
                + baseline_override + "-baseline"
    else:
        options = str(steps) + "steps_" \
                + baseline_override + "-baseline"

    if buffer_size != int(1e6): # if not default
        options += "_" + str(buffer_size) + "buffer_size"

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
    signature = inspect.signature(FiveXGoalSelection)
    default_goal_selection_params = {k: v.default
                                     for k, v in signature.parameters.items()
                                     if v.default is not inspect.Parameter.empty}
    merged = default_goal_selection_params | goal_selection_params
    conf_params["goal_selection_params"] = merged
    if baseline_override is not None:
        # if making a baseline, some parameters are ignored, remove from conf
        # for clarity
        del conf_params["goal_selection_params"]
        del conf_params["fixed_goal_fraction"]
    if baseline_override in ["base-rl", "curious"]:
        del conf_params["goal_weight"]
    with open(os.path.join(base_path, options, experiment, 'config.json'), 'w') as fp:
        json.dump(conf_params, fp, indent=4)

    # Initialize a training environment with default parameters
    #train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, vec_env_cls=SubprocVecEnv)
    train_env = gym.make(env_id, **env_params) #, render_mode='human')
    if train_seed is not None:
        train_env.reset(seed=train_seed)

    # wrap with goal conditioning and monitor wrappers
    if baseline_override in [None, "uniform-goal"]:
        # only use when training with goal
        train_env_goal = GoalWrapper(train_env,
                                     goal_weight=goal_weight,
                                     goal_range=goal_range)
                                     #intrinsic_curiosity_module=ICM(train_env,
                                     #                               device))
    else:
        train_env_goal = train_env # TODO handle that this name becomes missleading
    train_env = Monitor(train_env_goal, log_dir)
    #train_env = VecMonitor(train_env, log_dir)

    if env_id == "SparsePendulumEnv-v1":
        fixed_goal = lambda obs: np.array([1.0, 0.0, 0.0])
        coord_names = ["x", "y", "ang. vel."]
        algo = SAC
    elif env_id == "PathologicalMountainCar-v1.1":
        fixed_goal = lambda obs: np.array([-1.65, -0.02,])
        coord_names = ["xpos", "velocity"]
        # algo = DQN
        algo = DQNwithICM
        if baseline_override in ["base-rl", "curious"]:
            algo = DQN

    # Create log dir where evaluation results will be saved
    eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    # Separate evaluation env, with different parameters passed via env_kwargs
    # Eval environments can be vectorized to speed up evaluation.
    #eval_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
    #eval_env = GoalPendulumEnv(render_mode="human",
    #                           fixed_goal=np.array([1.0, 0.0, 0.0]))
    eval_env = gym.make(env_id, **env_params)#,render_mode="human")
    if eval_seed is not None:
        eval_env.reset(seed=eval_seed)
    # for eval we want to always evaluate with the goal at the top
    # if weight should be 0 (using real reward, or 1 using goal reward, or a mixture like
    # in training can be discussed)
    if baseline_override in [None, "uniform-goal"]:
        # only evaluate with goal when training with goal
        eval_env = GoalWrapper(eval_env,
                               goal_weight=0,
                               goal_range=goal_range,
                               goal_selection_strategies=fixed_goal)
    eval_env = Monitor(eval_env, eval_log_dir)

    # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
    # When using multiple training environments, agent will be evaluated every
    # eval_freq calls to train_env.step(), thus it will be evaluated every
    # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
    eval_callback = EvalCallback(eval_env,
                                best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir,
                                eval_freq=max(500 // n_training_envs, 1),
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False,
                                verbose=verbose)

    #check_env(train_env)

    if baseline_override in [None, "uniform-goal"]:
        # only use HER buffer when training with goal
        policy = "MultiInputPolicy"
        algo_kwargs = {"replay_buffer_class": HerReplayBuffer,
                       "replay_buffer_kwargs": dict(
                       n_sampled_goal=n_sampled_goal,
                       goal_selection_strategy="future",
                       copy_info_dict=True,), # TODO Turn off if not needed
                       }
    else:
        policy = "MlpPolicy"
        algo_kwargs = {}
    # TODO change here from Her buffer or just run without
    # without goal conditioning (but still truncated and hardstart)
    # could hack compute reward but that is probably just confusing
    model = algo(policy,
                train_env,
                **algo_kwargs,
                learning_starts=300,
                verbose=verbose,
                buffer_size=buffer_size,
                learning_rate=1e-3,
                gamma=0.95,
                batch_size=256,
                policy_kwargs=dict(net_arch=[256, 256, 256],),
                seed=policy_seed,
                device=device,
    )

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
                                                train_env_goal.targeted_goals,
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
        callback = CallbackList([eval_callback])

        #goal_selection_strategies = [train_env_goal.sample_obs_goal, fixed_goal]
        goal_selection_strategies = [goal_selection.select_goal_for_coverage, fixed_goal]
        goal_sel_strat_weight = [1-fixed_goal_fraction, fixed_goal_fraction]
        train_env_goal.set_goal_strategies(goal_selection_strategies, goal_sel_strat_weight)
        train_env_goal.print_setup()
    elif baseline_override == "uniform-goal":
        callback = eval_callback
        train_env_goal.set_goal_strategies([train_env_goal.sample_obs_goal])
        train_env_goal.print_setup()
    else:
        callback = eval_callback

    start_time = time.time()
    model.learn(steps, callback=callback, progress_bar=True)
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    #model.learn(steps, progress_bar=True)
    model_path = os.path.join(base_path, options, experiment, 'model')
    model.save(model_path)

    if baseline_override in [None, "uniform-goal"]:
        targeted_goals = np.stack(train_env_goal.targeted_goals)
        print(targeted_goals)
        utils.plot_targeted_goals(targeted_goals,
                                coord_names,
                                os.path.join(base_path, options, experiment))
        np.savetxt(os.path.join(base_path, options, experiment,"goals"), targeted_goals)

    # mark experiment complete
    with open(os.path.join(base_path, options, experiment, "completed.txt"), 'w') as fp:
        pass

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


if __name__ == '__main__':
    profile = False
    if profile:
        yappi.start()
    #experiments = ["test15",] #["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", ]
    #fixed_goal_fractions = [0.0,] #[0.0, 0.1, 0.5, 0.9, 1.0]
    #device = ["cpu", "cuda"]
    weights = [1, 1, 1, 1, 1]
    temp = utils.weight_combinations(weights, 1)
    temp.append([1, 1, 1, 1, 1])

    goal_conf_to_permute = {"exploit_dist": [0.2,],
                            "expand_dist": [0.01,],
                            "component_weights": [#[1, 1, 0, 0, 0], # try? [1, 0, 0, 5, 0], or so
                                                #   [1, 0, 1, 0, 0],
                                                #   [1, 0, 0, 1, 0],
                                                #   [1, 0, 0, 0, 1],
                                                #   [0, 1, 1, 0, 0],
                                                  [0, 1, 0, 1, 0],
                                                #   [0, 1, 0, 0, 1],
                                                #   [0, 0, 1, 1, 0],
                                                #   [0, 0, 1, 0, 1],
                                                #   [0, 0, 0, 1, 1],
                                                 ],
                            # "component_weights": utils.weight_combinations(weights, 2),
                            # "component_weights": temp,
                            "steps_halflife": [500,],
                            "escalate_exploit": [True],
                            "norm_comps": [True],
                            }
    env_params = {#"harder_start": [0.1],
                  "terminate": [True]
                  }

    params_to_permute = {"experiments": [1],
                         "env_id": ["PathologicalMountainCar-v1.1",], # "PathologicalMountainCar-v1" # "SparsePendulumEnv-v1" # "Pendulum-v1" "MountainCarContinuous-v0"
                         "fixed_goal_fraction": [0.0],
                         "device": ["cuda"],
                         "steps": [500000],
                         "goal_weight": [1.0],
                         "goal_range": [0.1],
                         "goal_selection_params": named_permutations(goal_conf_to_permute),
                         "env_params": named_permutations(env_params),
                         "buffer_size": [20000],
                         "baseline_override": [None] #["base-rl", "uniform-goal"]  # should be if not doing baseline None
                         }

    base_path = "./temp/wrapper/"

    experiment_list = named_permutations(params_to_permute)

    # run test runs first to confirm needed number of experiments
    full_experiment_list = []
    total_planned = 0
    total_found = 0
    for conf in experiment_list:
        completed_exps = train(base_path = base_path,
                               verbose = 0,
                               test_needed_experiments = True,
                               **conf)
        total_planned += conf["experiments"]
        total_found += min(len(completed_exps), conf["experiments"])
        if len(completed_exps) < conf["experiments"]:
            full_experiment_list += [conf]*(conf["experiments"]-len(completed_exps))

    print(f"Out of {total_planned} planned experiments, {total_found} were already completed")
    print(f"{len(full_experiment_list)} experiments queued up to be run")
    total_time = 0 # in seconds
    for i, conf in enumerate(full_experiment_list):
        print("Training with configuration: " + str(conf))
        start_time = time.time()

        train(base_path = base_path,
              verbose = 0,
              **conf)

        time_used = time.time() - start_time
        total_time += time_used
        part_time = time.strftime('%H:%M:%S', time.gmtime(time_used))
        print(f"--- latest experiment took {part_time} ---")
        total_estimate = total_time/(i+1) * len(full_experiment_list)
        completed_time = time.strftime('%H:%M:%S', time.gmtime(total_time))
        total_time_estimate = time.strftime('%H:%M:%S', time.gmtime(total_estimate))
        print(f"--- {i+1}/{len(full_experiment_list)} experiments have been " \
            + f"completed in {completed_time}/{total_time_estimate} (total time estimated) ---")

    if profile:
        yappi.stop()
        yappi.get_func_stats().save("./test3.pstats",type="pstat")

    # for conf in product(fixed_goal_fractions, experiments):
    #     print("Training with configuration: " + str(conf))
    #     train(fixed_goal_fraction = conf[0],
    #             experiment=conf[1],
    #             steps=20000,
    #             goal_weight=1.0,
    #             device="cuda"
    #             )