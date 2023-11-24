import gymnasium as gym
import numpy as np

from gymnasium import error, logger
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.utils import flatten, flatdim
from gymnasium import spaces


# lets make a non-vecotrized version first

class GoalWrapper(
    #gym.Wrapper[ObsType, ActType, ObsType, ActType],
    gym.Wrapper,
    gym.utils.RecordConstructorArgs
):
    def __init__(
            self,
            env: gym.Env,
            intrinsic_weight: float=0.5,
            reward_func = None, # TODO make type hint with Callable
            goal_selection_strat = None, # TODO make type hint with Callable
            # env: gym.Env[ObsType, ActType],
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            intrinsic_weight=intrinsic_weight,
        )
        gym.Wrapper.__init__(self, env)

        self.intrinsic_weight = intrinsic_weight
        # for now this only supports flattenable observation spaces
        assert self.env.observation_space.is_np_flattenable
        # TODO this supports inputing a reward function from the outside, but we should
        # proabaly also support selecting different strategies in this code with sting arguments
        if reward_func is None:
            self.compute_reward = self.flatten_norm_reward
        else:
            self.compute_reward = reward_func
        if goal_selection_strat is None:
            self.select_goal = self.sample_obs_goal
        else:
            self.select_goal = goal_selection_strat

        new_observation_space = spaces.Dict({"observation": self.env.observation_space,
                                             "achieved_goal": self.env.observation_space,
                                             "desired_goal": self.env.observation_space})

        # TODO determine if I should replace this with inheriting TransformObservation
        # and calling its constructor here
        # If so I should probably do the same for reward
        self.observation_space = new_observation_space
        self.goal_dim = flatdim(self.observation_space["desired_goal"])

    def step(
            self,
            action
            ):
            #) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]: # TODO remove or fix imports of types and such
        obs, reward, terminated, truncated, info = self.env.step(action)

        # get goal conditioned reward
        inreward = self.compute_reward(obs, self.goal, info)
        iw = self.intrinsic_weight
        # output weighted average of rewards
        totreward = iw*inreward + (1-iw)*reward

        # save separate rewards in info
        assert info.get("exreward") == None
        assert info.get("inreward") == None
        info["exreward"] = reward
        info["inreward"] = inreward

        return self._get_obs(obs), totreward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, reset_info = self.env.reset(**kwargs)
        self.goal = self.select_goal()

        return self._get_obs(obs), reset_info

    def sample_obs_goal(self):
        obs_space = self.env.observation_space
        
        # first trivial goal selection (uniform/default in obs space)
        return obs_space.sample()

    def _get_obs(self, obs):
        # current version assumes goals in obs space, expand this when that is not 
        # always the case anymore
        return {"observation": obs, "achieved_goal": obs, "desired_goal": self.goal}
        
    def flatten_norm_reward(self,
                       achieved_goal,
                       desired_goal,
                       info,
                       ):
        # TODO we need to handle all types of obs spaces here to be able to compare
        # and check "distance" from goal. For now requires flattenable space
        # in the future it should also be possible to give weights to certain dimensions
        # especially by normalizing it by the size of each obs space component
        flat_achi = flatten(self.env.observation_space, achieved_goal)
        flat_goal = flatten(self.env.observation_space, desired_goal)
        if achieved_goal.ndim == 2: # handle batch
            flat_achi = flat_achi.reshape(-1, self.goal_dim)
            flat_goal = flat_goal.reshape(-1, self.goal_dim)
        distance = np.linalg.norm(flat_achi - flat_goal, axis=-1) # TODO not sure about axis here
        # for now returning dense reward, hard to extimate generic cutoff value
        return np.exp(-distance)


