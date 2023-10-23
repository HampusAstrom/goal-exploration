import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from goal_pendulum import GoalPendulumEnv

#env_id = "Pendulum-v1"
n_training_envs = 1
n_eval_envs = 5

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# TODO replace with command line argument

# Create log dir
log_dir = os.path.join("./runs/pendulum/", "goal/sparse/", "exp1", "train_logs")
os.makedirs(log_dir, exist_ok=True)

# Initialize a vectorized training environment with default parameters
#train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
#train_env = GoalPendulumEnv(render_mode="human")
train_env = gym.make("GoalPendulumEnv-v0", reward_type="sparse")
train_env = Monitor(train_env, log_dir)

# Create log dir where evaluation results will be saved
eval_log_dir = os.path.join("./runs/pendulum/", "goal/sparse/", "exp1", "eval_logs")
os.makedirs(eval_log_dir, exist_ok=True)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
#eval_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)
#eval_env = GoalPendulumEnv(render_mode="human",
#                           fixed_goal=np.array([1.0, 0.0, 0.0]))
eval_env = gym.make("GoalPendulumEnv-v0",
                    render_mode="human",
                    fixed_goal=np.array([1.0, 0.0, 0.0]),
                    reward_type="sparse")
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

model = SAC("MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            learning_starts=500,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
)
#model.learn(5000, callback=eval_callback, progress_bar=True)
model.learn(100000, progress_bar=True)