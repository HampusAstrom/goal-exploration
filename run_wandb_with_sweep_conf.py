import numpy as np
import os
import argparse
from functools import partial
from pprint import pprint

import wandb
from wandb.apis.public.runs import Runs
import wandb.sdk
import wandb.sdk.launch
import wandb.sdk.launch.sweeps
import wandb.sdk.launch.sweeps.utils

from goal_wrapper_run import train, confirm
import utils

def nested_dict2sweep_params(dct):
    ret_dict = {}
    for key, val in dct.items():
        if isinstance(val, dict):
            ret_dict[key] = {"parameters": nested_dict2sweep_params(val)}
        elif isinstance(val, list):
            # if only one val, set as value
            if len(val) == 1:
                ret_dict[key] = {"value": val[0]}
            elif len(val) > 1:
                ret_dict[key] = {"values": val}
    return ret_dict

def get_past_runs(sweep_ids = None,
                  entity = "hampus-astrom-lund-university",
                  project = "MountainCar-v0|max_episode_steps[1000]|test_run"):
    api = wandb.Api()
    # filters = {"state": "finished"}
    # all_groups_runs = api.runs(entity + "/" + project,
    #                            filters = filters,
    #                            per_page = 50,
    #                            )
    sweeps = api.project(name=project, entity=entity).sweeps()

    ids_per_sweep = {}
    # Iterate over sweeps and print details
    for sweep in sweeps:
        if sweep_ids is None or sweep.id in sweep_ids:
            print(f"Sweep name: {sweep.name}")
            print(f"Sweep ID: {sweep.id}")
            print(f"Sweep URL: {sweep.url}")
            print("----------")

            filters = {"state": "finished"}
            runs = sweep.runs
            ids = []
            for run in runs:
                if run.state == "finished":
                    ids.append(run.id)
            ids_per_sweep[sweep.id] = ids

    return ids_per_sweep


def main():
    # as a test I don't trust it, so make a partial func to give
    func = partial(train,
                   )

    with wandb.init() as run:
        conf = run.config
        pprint(conf)

        train(base_path=base_path,
              verbose=0,
              eval_seed=2,
              test_run=False, # replace after all works
              **conf,
              )

# TODO move actual config to own file, and make a default one to point to if not
# provided via command line argument

goal_conf_to_permute = dict(
    grid_size=[100],
    fraction_random=[0.01], # 0.01
    # target_success_rate=[0.75], # turn on if intermediate instead of novelty # 0.75
    dist_decay=[2],
)

env_id = "MountainCar-v0" # "Acrobot-v1" # "MountainCar-v0" "CliffWalking-v0" "PathologicalMountainCar-v1.1" # "FrozenLake-v1" "PathologicalMountainCar-v1.1" "SparsePendulumEnv-v1"
# TODO get updated gym envs with cliffwalking v1

env_params_override = dict(
    max_episode_steps=[1000]
)

policy_kwargs_to_permute=dict(
    net_arch=[[80],] # [80], [32, 32],[128, 128], [64, 64], [64, 64, 64],
)

algo_kwargs_to_permute = dict(
    policy_kwargs=policy_kwargs_to_permute,
    learning_rate=[1e-3,], #1e-3
    batch_size=[128], # 256
    buffer_size=[50_000,],
    target_update_interval=[100], # 10,100,200,500,1000
    gamma=[0.99,], # 0.95, 0.99,0.999
    exploration_fraction=[0.25,], # 0.5,0.2,0.1
    exploration_initial_eps=[1.0], # 1,0.1,0.01,
    exploration_final_eps=[0.001],
    train_freq=[1,], # 1,3,10,30
    double_dqn=[False, True],
    learning_starts=[5_000], # 500,1000,3000,10_000
    #tau=[0.2,0.3], # 1.0, 0.5, 0.1
)

params_to_permute = dict(
    experiments=[100], # temp, doesn't seem to happen, so only to not make it term before from it
    env_id=[env_id],
    fixed_goal_fraction=[0.0],
    device=["cpu"],
    steps=[1_000_000], # 10_000_000 pmc default
    goal_weight=[1.0],
    goal_range=[0.0001],
    goal_selection_params=goal_conf_to_permute,
    env_params_override=env_params_override,
    algo_kwargs=algo_kwargs_to_permute,
    after_goal_success=["reselect"], # "exact_goal_match_reward" "term", "reselect", "local-reselect" # only applies to train, eval terms
    # TODO reward_func is replaced with a pure term or reselect param
    # but we also need one for reward eval and handle it's relation to
    # goal selection methods
    baseline_override=["base-rl"], # ["base-rl", "uniform-goal", "grid-novelty"] [None]  # should be if not doing baseline None
    range_as_goal=[False], # only works with uniform goal selection for now
    #algo_override=[PPO],
    n_sampled_goal=[4],
    t2g_ratio=[(1, "raw")], #first part ratio, last "raw" if raw or "her" on top of her ratio
)

default_parameters=nested_dict2sweep_params(params_to_permute)

parameters=utils.deepcopy(default_parameters)
# overrides with ranges and such here
# non-deep copies for easy access past extra layers:
algo_kwargs=parameters["algo_kwargs"]["parameters"]
policy_kwargs=algo_kwargs["policy_kwargs"]["parameters"]
#env_params_override=parameters["env_params_override"]["parameters"] # don't touch for now
#goal_selection_params=parameters["goal_selection_params"]["parameters"] # shouldn't matter for base

# just remember to replace entire dict entry under key, as there is a value/values there now
policy_kwargs["net_arch"] = dict(values=[[80], [32, 32], [64, 64], [128, 128],])
algo_kwargs["exploration_fraction"] = dict(min=0.0, max=1.0, distribution="uniform")
algo_kwargs["exploration_initial_eps"] = dict(min=0.0, max=1.0, distribution="uniform")
algo_kwargs["exploration_final_eps"] = dict(min=0.0, max=1.0, distribution="uniform")
algo_kwargs["learning_rate"] = dict(min=5e-5, max=5e-3,)
algo_kwargs["batch_size"] = dict(values=[64, 128, 256, 512])
algo_kwargs["buffer_size"] = dict(min=5_000, max=1_000_000)
algo_kwargs["train_freq"] = dict(min=1, max=300) # needs better prior or dist
algo_kwargs["learning_starts"] = dict(min=1000, max=50_000)
algo_kwargs["target_update_interval"] = dict(min=10, max=5000)
#algo_kwargs["tau"] = dict(min=0.1, max=1.0)
# try tau later too

base_path = "./output/wrapper/"

# TODO sort up loose params here
window = 10

sweep_conf=dict(
    method="bayes",
    metric=dict(
        name="meta_eval/meta_value", # f"meta_eval/avg_reward_for_{window}_evals_best",
        goal="miminize", # maximize
        # could add target here
    ),
    # TODO insert params_to_permute here, converted to non-single values
    # as inital try of conf
    parameters=parameters,
)

# TODO TODO TODO TODO TODO TODO
# break out all param pre-processing from goal_wrapper_run to separate funtion
# that can be called here and in goal_wrapper_run
# then we can get it to run correctly

# TODO also try to see if I can update and continnue sweep instead of restarting?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_run', action='store_true')
    parser.add_argument('-n', '--no_prior_runs', action='store_true')
    args = parser.parse_args()

    pprint(sweep_conf)

    wandb.login()

    proj_name = env_id
    # if env params changed, add to proj name, as it different task then
    if env_params_override is not None:
        proj_name += "|" + utils.dict2string(env_params_override)

    if args.test_run:
        proj_name += "|test_run"

    # TODO get and add list of prior runs here
    prior_runs = []
    if not args.no_prior_runs:
        prior_run_ids = get_past_runs()
        pprint(prior_run_ids)
        for key, val in prior_run_ids.items():
            prior_runs += val

    pprint(f"Project name for sweep: {proj_name}")
    pprint(prior_runs)

    if not confirm():
        exit()

    sweep_id = wandb.sweep(sweep=sweep_conf,
                           project=proj_name,
                           prior_runs=prior_runs,
                           )

    print(f"Sweep id: {sweep_id}")

    # wandb.agent(sweep_id,
    #             function=main,)