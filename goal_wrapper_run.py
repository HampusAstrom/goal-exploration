import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import inspect
import json
import utils

from stable_baselines3 import SAC, HerReplayBuffer, DQN, PPO
from stable_baselines3.dqn import DQNwithICM
from stable_baselines3.sac import SACwithICM
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

#import imageio

import time
import yappi

from goal_wrapper import GoalWrapper, FiveXGoalSelection, OrderedGoalSelection, GridNoveltySelection, FixedGoalSelection

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
          reward_func: str = "term",
          env_params: dict = None,
          env_id = "PathologicalMountainCar-v1.1",
          buffer_size = int(1e6),
          baseline_override = None, # alternatives: "base-rl", "curious", "uniform-goal"
          verbose = 0,
          test_needed_experiments = False,
          algo_override = None,
          n_sampled_goal = 4, # Create 4 artificial transitions per real transition
          t2g_ratio = (1, "raw"), # train2gather ratio, first ratio, last "her" if mult with her_factor
         ):

    base_path = os.path.join(base_path,env_id)
    n_training_envs = 1
    n_eval_envs = 5

    if env_id == "PathologicalMountainCar-v1.1" or \
        env_id == "CliffWalking-v0":
        eval_freq = 50000 #50000 for patho MC and Cliffwalk (high freq 10000), 10000 frozen
    elif env_id == "FrozenLake-v1":
        eval_freq = 10000
    else:
        raise "Error, set eval_freq for this env"

    # Collect variables to store in json before cluttered
    conf_params = locals()

    if baseline_override in [None, "grid-novelty"]:
        options = str(steps) + "steps_" \
                + str(goal_weight) + "goalrewardWeight_" \
                + str(fixed_goal_fraction) + "fixedGoalFraction_" \
                + str(goal_range) + "goal_range_" \
                + str(reward_func) + "-reward_func" \

    if baseline_override in [None, "grid-novelty"]:
        for key, val in goal_selection_params.items():
            if type(val) is list:
                options += "_[" + ",".join(str(v) for v in val) + "]" + key
            elif val is True:
                options += "_" + key
            else:
                options += "_" + str(val) + key
    elif baseline_override == "uniform-goal":
        options = str(steps) + "steps_" \
                + str(goal_weight) + "goalrewardWeight_" \
                + str(goal_range) + "goal_range_" \
                + str(reward_func) + "-reward_func_" \
                + baseline_override + "-baseline"
    if baseline_override in ["base-rl", "curious"]:
        options = str(steps) + "steps_" \
                + baseline_override + "-baseline"

    if buffer_size != int(1e6): # if not default
        options += "_" + str(buffer_size) + "buffer_size"

    if n_sampled_goal != 4 and baseline_override not in ["base-rl"]:
        options += "_" + str(n_sampled_goal+1) + "her_factor"

    if t2g_ratio != (1, "raw"):
        if t2g_ratio[1] == "her":
            options += "_herX" + str(t2g_ratio[0]) + "t2g"
        else:
            options += "_" + str(t2g_ratio[0]) + "t2g"

    if algo_override is not None:
        options += "_" + str(algo_override.__name__)

    if t2g_ratio[1] == "her":
        t2g = t2g_ratio[0]*(n_sampled_goal+1)
    else:
        t2g = t2g_ratio[0]

    # TODO hard coded options addition
    #options += "_256-256-256-256-256net"
    #options += "_finetune_at0.7_1.0of_time"#"_no_hindsight_then"
    #options += "_gradually_reduce_hindsight_from_0.8_to_0"
    #options += "_her_episode_gradually_reduce_hindsight_from_0.8_to_0"
    #options += "_batch_size512_train_freq512_lr_1e-3_4x256resblock-x4net_exploration_fraction0.5_more_novelty_focus"
    #options += "_batch_size512_train_freq512_lr_1e-3_3x256net"#_more_novelty_focus"
    options += "_batch_size512_lr_1e-3_2x256net" # _0.1-0.01_epsilonexplore #4x256resblock-x4net" #_no_term_in_real"

    print(options)

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
    conf_params["reward_func"] = reward_func
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
                                     reward_func=reward_func)
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
        fixed_goal = lambda obs: np.array([-1.60, 0.00,])
        few_goals = [fixed_goal,
                     lambda obs: np.array([0.5, 0.00,])]
        many_goals = few_goals.copy()
        many_goals += [lambda obs: np.array([-0.5, 0.04,]),
                       lambda obs: np.array([-0.5, -0.04,]),
                       lambda obs: np.array([-0.7, 0.0,])]
        max_goals = [[-1.60, 0.00,],
                     [0.5, 0.00,],
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
        coord_names = ["xpos", "velocity"]
        # algo = DQN
        algo = DQNwithICM
        if baseline_override in ["base-rl", "curious"]: # TODO this looks wrong, I don't think i use "curious" yet
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
    if algo_override is not None:
        algo = algo_override

    def setup_evalcallbacks_from_goal_list(goals, freq=eval_freq, n_eval_episodes=1):

        eval_callback_list = []
        for goal in goals:
            eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs_" + str(goal))
            os.makedirs(eval_log_dir, exist_ok=True)

            eval_env = gym.make(env_id, **env_params)
            if eval_seed is not None:
                eval_env.reset(seed=eval_seed)
            goal_selector = FixedGoalSelection(goal)
            goal_selection = goal_selector.select_goal
            goal_weight=1

            eval_env_goal = GoalWrapper(eval_env,
                                goal_weight=goal_weight,
                                goal_range=goal_range,
                                goal_selection_strategies=goal_selection,
                                reward_func=reward_func) # "term" TODO OBS CHANGED HERE!!!!!!!!!!!!!!
            eval_env = Monitor(eval_env_goal, eval_log_dir)

            eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=eval_log_dir,
                                    log_path=eval_log_dir,
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
        if eval_type == "many":
            goal_selector = OrderedGoalSelection(many_goals)
            goal_selection = goal_selector.select_goal
            goal_weight=1
            n_eval_episodes=goal_selector.len
        if eval_type == "fixed":
            goal_selection = fixed_goal
            goal_weight=0
            n_eval_episodes=5


        # for eval we want to always evaluate with the goal at the top
        # if weight should be 0 (using real reward, or 1 using goal reward, or a mixture like
        # in training) can be discussed
        # TODO replace all these checks with non-listed logic
        if baseline_override in [None, "uniform-goal", "grid-novelty"]:
            # only evaluate with goal when training with goal
            eval_env_goal = GoalWrapper(eval_env,
                                goal_weight=goal_weight,
                                goal_range=goal_range,
                                goal_selection_strategies=goal_selection,
                                reward_func=reward_func) # "term" TODO OBS CHANGED HERE!!!!!!!!!!!!!!
        else:
            eval_env_goal = eval_env # TODO handle that this name becomes missleading
        eval_env = Monitor(eval_env_goal, eval_log_dir)

        # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
        # When using multiple training environments, agent will be evaluated every
        # eval_freq calls to train_env.step(), thus it will be evaluated every
        # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
        # TODO set identical env initiations in eval
        eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=eval_log_dir,
                                    log_path=eval_log_dir,
                                    eval_freq=max(eval_freq // n_training_envs, 1),
                                    n_eval_episodes=n_eval_episodes,
                                    deterministic=True,
                                    render=False,
                                    verbose=verbose,
                                    seed=eval_seed)

        return eval_callback

    eval_callback = setup_eval("fixed")
    # TODO replace all these checks with non-listed logic
    if baseline_override != "base-rl" and True:
        clist = setup_evalcallbacks_from_goal_list(max_goals, n_eval_episodes=1)
        clist.append(eval_callback)
        eval_callbacks = clist
    elif baseline_override in [None, "uniform-goal", "grid-novelty"]:
        eval_callback_few = setup_eval("few")
        eval_callback_many = setup_eval("many")
        eval_callbacks = [eval_callback,
                          eval_callback_few,
                          eval_callback_many]
        # TODO this is a bit inefficient, we could gain speed
    else:
        eval_callbacks = [eval_callback]

    eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    #check_env(train_env)

    # TODO replace all these checks with non-listed logic
    if baseline_override in [None, "uniform-goal", "grid-novelty"]:
        # only use HER buffer when training with goal
        policy = "MultiInputPolicy"
        algo_kwargs = {"replay_buffer_class": HerReplayBuffer,
                       "replay_buffer_kwargs": dict(
                       n_sampled_goal=n_sampled_goal,
                       goal_selection_strategy="future",
                       copy_info_dict=True, # TODO Turn off if not needed
                       terminate_at_goal=True), # TODO make dependant on goal eval strat
                       }
        # if env_id == "CliffWalking-v0":
        #     algo_kwargs["replay_buffer_kwargs"]["handle_timeout_termination"] = True
    else:
        policy = "MlpPolicy"
        algo_kwargs = {}
    # TODO change here from Her buffer or just run without
    # without goal conditioning (but still truncated and hardstart)
    # could hack compute reward but that is probably just confusing
    batch_size = 512
    model = algo(policy,
                train_env,
                **algo_kwargs,
                learning_starts=300,
                verbose=verbose,
                buffer_size=buffer_size,
                learning_rate=1e-3,
                gamma=0.95,
                batch_size=batch_size,
                train_freq=int(batch_size/t2g),
                #exploration_fraction=1.0,
                #target_update_interval=1000,
                #gradient_steps=-1,
                #policy_kwargs=dict(net_arch=[256, 256, 256],),
                #policy_kwargs=dict(net_arch=[128, 128, 128],),
                policy_kwargs=dict(net_arch=[256, 256],),
                #policy_kwargs=dict(net_arch=[64, 64],),
                #policy_kwargs=dict(net_arch=[256, 256, 256, 256, 256, 256, 256, 256, 256],),
                #policy_kwargs=dict(net_arch={"res-block": [256, 256, 256, 256], "num_blocks": 4},),
                seed=policy_seed,
                device=device,
                tensorboard_log=log_dir,
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
        callback = eval_callbacks

        #goal_selection_strategies = [train_env_goal.sample_obs_goal, fixed_goal]
        goal_selection_strategies = [goal_selection.select_goal_for_coverage, fixed_goal]
        goal_sel_strat_weight = [1-fixed_goal_fraction, fixed_goal_fraction]
        train_env_goal.set_goal_strategies(goal_selection_strategies, goal_sel_strat_weight)
        train_env_goal.print_setup()
    elif baseline_override == "uniform-goal":
        callback = eval_callbacks
        train_env_goal.set_goal_strategies([train_env_goal.sample_obs_goal])
        train_env_goal.print_setup()
    elif baseline_override == "grid-novelty":
        callback = eval_callbacks
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
        callback = eval_callbacks

    print(model.policy)

    start_time = time.time()
    model.learn(steps, callback=CallbackList(callback), progress_bar=True)
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    #model.learn(steps, progress_bar=True)
    model_path = os.path.join(base_path, options, experiment, 'model')
    model.save(model_path)

    if baseline_override in [None, "uniform-goal", "grid-novelty"]:
        targeted_goals = np.stack(train_env_goal.targeted_goals)
        initial_targeted_goals = np.stack(train_env_goal.initial_targeted_goals)
        successful_goals = np.stack(train_env_goal.successful_goals)
        successful_goal_index = np.stack(train_env_goal.successful_goal_index)
        np.savetxt(os.path.join(base_path, options, experiment,"goals"), targeted_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"initial_targeted_goals"), initial_targeted_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"successful_goal_spread"), successful_goals)
        np.savetxt(os.path.join(base_path, options, experiment,"successful_goal_index"), successful_goal_index)

        if len(train_env_goal.local_targeted_goals) > 0:
            local_targeted_goals = np.stack(train_env_goal.local_targeted_goals)
            np.savetxt(os.path.join(base_path, options, experiment,"reselect_goal_spread"), local_targeted_goals)

        if len(targeted_goals.shape) > 1: # requires goals tracked in more than one dim to plot this way
            utils.plot_targeted_goals(targeted_goals,
                                    coord_names,
                                    os.path.join(base_path, options, experiment))
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
    profile = False
    if profile:
        yappi.start()
    #experiments = ["test15",] #["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", ]
    #fixed_goal_fractions = [0.0,] #[0.0, 0.1, 0.5, 0.9, 1.0]
    #device = ["cpu", "cuda"]
    weights = [1, 1, 1, 1, 1]
    temp = utils.weight_combinations(weights, 1)
    temp.append([1, 1, 1, 1, 1])

    # goal_conf_to_permute = {"exploit_dist": [0.2,],
    #                         "expand_dist": [0.01,],
    #                         "component_weights": [[0, 0, 0, 0, 1],
    #                                             #   [1, 1, 1, 1, 0],
    #                                             #   [0, 1, 1, 1, 0],
    #                                             #   [1, 1, 0, 1, 0],
    #                                             #   [1, 0, 0, 1, 0],
    #                                             #   [1, 1, 0, 1, 0],
    #                                             #   [0, 0, 0, 1, 0],
    #                                             #   [1, 0, 0, 0, 1],
    #                                             #   [1, 0, 0, 1, 0],
    #                                             #   [0, 0, 0, 1, 1],
    #                                             #   [1, 0, 0, 0, 0],
    #                                             #   [0, 0, 0, 0, 1],
    #                                              ],
    #                         # "component_weights": utils.weight_combinations(weights, 2),
    #                         # "component_weights": temp,
    #                         "steps_halflife": [500,],
    #                         "escalate_exploit": [True],
    #                         "norm_comps": [True],
    #                         }
    goal_conf_to_permute = {"grid_size": [100],
                            "fraction_random": [0.1],
                            #"target_success_rate": [0.75], # turn on if intermediate instead of novelty # 0.75
                            "dist_decay": [2],
                            }

    pend_env_params = {"harder_start": [0.1]}
    pathmc_env_params = {"terminate": [True]}
    frozen_env_params = {"is_slippery": [True]}
    cliffwalker_env_params = {"max_episode_steps": [300]} # override to play nice with HER #{"is_slippery": [False]}

    env_id = "PathologicalMountainCar-v1.1" # "CliffWalking-v0" "PathologicalMountainCar-v1.1" # "FrozenLake-v1" "PathologicalMountainCar-v1.1" "SparsePendulumEnv-v1"
    # TODO get updated gym envs with cliffwalking v1

    if env_id == "SparsePendulumEnv-v1":
        env_params = pend_env_params
    elif env_id == "PathologicalMountainCar-v1.1":
        env_params = pathmc_env_params
    elif env_id == "FrozenLake-v1":
        env_params = frozen_env_params
    elif env_id == "CliffWalking-v0":
        env_params = cliffwalker_env_params
    else:
        print("env without env params?")
        exit()
    # env_params = {#"harder_start": [0.1], # pendulum
    #               "terminate": [True] # patho mc
    #               #"is_slippery": [True]
    #               }

    params_to_permute = {"experiments": [2],
                         "env_id": [env_id],
                         "fixed_goal_fraction": [0.0],
                         "device": ["cuda"],
                         "steps": [10_000_000], # 10_000_000 pmc default
                         "goal_weight": [1.0],
                         "goal_range": [0.01],
                         "goal_selection_params": named_permutations(goal_conf_to_permute),
                         "env_params": named_permutations(env_params),
                         "reward_func": ["reselect"], # "exact_goal_match_reward" "term", "reselect", "local-reselect" # only applies to train, eval terms
                         "buffer_size": [10_000_000],
                         "baseline_override": ["uniform-goal"], # ["base-rl", "uniform-goal", "grid-novelty"] [None]  # should be if not doing baseline None
                         #"algo_override": [PPO],
                         "n_sampled_goal": [4],
                         "t2g_ratio": [(1, "her")], #first part ratio, last "raw" if raw or "her" on top of her ratio
                         }

    base_path = "./output/wrapper/"

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
    if not confirm():
        exit()
    total_time = 0 # in seconds
    for i, conf in enumerate(full_experiment_list):
        print("Training with configuration: " + str(conf))
        start_time = time.time()

        train(base_path = base_path,
              verbose = 0,
              eval_seed = 2, # seeds 2,3 on the easeir side in path MC, seed 0 at the bottom, 4 on hard side
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
        yappi.get_func_stats().save("./profile1milStep_10kBuffer.pstats",type="pstat")

    # for conf in product(fixed_goal_fractions, experiments):
    #     print("Training with configuration: " + str(conf))
    #     train(fixed_goal_fraction = conf[0],
    #             experiment=conf[1],
    #             steps=20000,
    #             goal_weight=1.0,
    #             device="cuda"
    #             )