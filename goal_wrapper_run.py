import os
import gymnasium as gym
import numpy as np
from itertools import product

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

import imageio

from goal_wrapper import GoalWrapper

def train(base_path: str = "./temp/wrapper/pendulum/",
          steps: int = 10000,
          experiment: str = "exp1",
          intrinsic_weight: float = 0.5,
          eval_seed: int = None,
          train_seed: int = None,
          policy_seed: int = None,
         ):

    env_id = "Pendulum-v1" # "MountainCarContinuous-v0"
    n_training_envs = 1
    n_eval_envs = 5

    # Create 4 artificial transitions per real transition
    n_sampled_goal = 4

    options = str(steps) + "steps_" + str(intrinsic_weight) + "inrewardweight"

    # Create log dir
    log_dir = os.path.join(base_path, options, experiment, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize a training environment with default parameters
    train_env = gym.make(env_id)
    if train_seed is not None:
        train_env.reset(seed=train_seed)

    # wrap with goal conditioning and monitor wrappers
    train_env = GoalWrapper(train_env, intrinsic_weight=intrinsic_weight)
    train_env = Monitor(train_env, log_dir)

    # Create log dir where evaluation results will be saved
    eval_log_dir = os.path.join(base_path, options, experiment, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    # Separate evaluation env, with different parameters passed via env_kwargs
    # Eval environments can be vectorized to speed up evaluation.
    #eval_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
    #eval_env = GoalPendulumEnv(render_mode="human",
    #                           fixed_goal=np.array([1.0, 0.0, 0.0]))
    eval_env = gym.make(env_id,
                        render_mode="human")
    if eval_seed is not None:
        eval_env.reset(seed=eval_seed)
    eval_env = GoalWrapper(eval_env, intrinsic_weight=intrinsic_weight)
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
                                render=True)

    #check_env(train_env)

    # TODO change here from Her buffer or just run without
    # without goal conditioning (but still truncated and hardstart)
    # could hack compute reward but that is probably just confusing
    model = SAC("MultiInputPolicy",
                train_env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=n_sampled_goal,
                    goal_selection_strategy="future",
                    copy_info_dict=True,
                ),
                learning_starts=300,
                verbose=1,
                buffer_size=int(1e6),
                learning_rate=1e-3,
                gamma=0.95,
                batch_size=256,
                policy_kwargs=dict(net_arch=[256, 256, 256],),
                seed=policy_seed,
    )
    model.learn(steps, callback=eval_callback, progress_bar=True)
    #model.learn(5000, progress_bar=True)
    model_path = os.path.join(base_path, options, experiment, 'model')
    model.save(model_path)

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

if __name__ == '__main__':
    experiments = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", ]
    
    for conf in product(experiments):
        train(experiment=conf[0],
              steps=20000,
              intrinsic_weight=1.0,
              )
        print(conf)