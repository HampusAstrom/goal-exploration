import gymnasium as gym
import numpy as np
import scipy

from gymnasium import error, logger
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.utils import flatten, flatdim
from gymnasium import spaces


# TODO make class? for submitting multiple goal selection strategies with weights for each
# to be used a fraction of the time in proportion to the weight
# we can then also use this to input final goal when running evaluation
# we should probably also make goal selection strategy into something that can be changed
# without remaking wrapper

# TODO write method for printing current setup before starting training

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
            goal_selection_strategies = None, # TODO make type hint with Callable
            goal_sel_strat_weight = None,
            # env: gym.Env[ObsType, ActType],
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            intrinsic_weight=intrinsic_weight,
        )
        gym.Wrapper.__init__(self, env)

        assert intrinsic_weight >=0 and intrinsic_weight <= 1
        self.intrinsic_weight = intrinsic_weight
        # for now this only supports flattenable observation spaces
        assert self.env.observation_space.is_np_flattenable
        # TODO this supports inputing a reward function from the outside, but we should
        # proabaly also support selecting different strategies in this code with sting arguments
        if reward_func is None:
            self.compute_reward = self.flatten_norm_reward
        else:
            self.compute_reward = reward_func

        self.selection_strategies = None
        self.strat_weights = None
        if goal_selection_strategies is None:
            self.select_goal = self.sample_obs_goal
        elif isinstance(goal_selection_strategies, list):
            self.set_goal_strategies(goal_selection_strategies, goal_sel_strat_weight)
        else:
            self.select_goal = goal_selection_strategies

        new_observation_space = spaces.Dict({"observation": self.env.observation_space,
                                             "achieved_goal": self.env.observation_space,
                                             "desired_goal": self.env.observation_space})

        # TODO determine if I should replace this with inheriting TransformObservation
        # and calling its constructor here
        # If so I should probably do the same for reward
        self.observation_space = new_observation_space
        self.goal_dim = flatdim(self.observation_space["desired_goal"])

        # create variables that track seen and targeted goals
        # seen goals can maybe just we replay buffer from policy algorithm
        # TODO replace with flexible solution
        self.seen_goals = []
        self.targeted_goals = []

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

        self.seen_goals.append(obs)

        return self._get_obs(obs), totreward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, reset_info = self.env.reset(**kwargs)
        self.goal = self.select_goal(obs) # goal selection can require obs info

        self.targeted_goals.append(self.goal)

        return self._get_obs(obs), reset_info

    def _get_obs(self, obs):
        # current version assumes goals in obs space, expand this when that is not
        # always the case anymore
        return {"observation": obs, "achieved_goal": obs, "desired_goal": self.goal}

    def set_goal_strategies(self, goal_selection_strategies, goal_sel_strat_weight=None):
        # if list of selection strategies but not weights, uniform is assumed
        self.selection_strategies = goal_selection_strategies
        if goal_sel_strat_weight is None:
            self.strat_weights = [1]*len(goal_selection_strategies)
        else:
            self.strat_weights = goal_sel_strat_weight
        self.select_goal = self.multi_strat_goal_selection

    def print_setup(self):
        print()
        print("Goal wrapper setup")
        print("intrinsic_weight: " + str(self.intrinsic_weight))
        print("goal selection method: " + str(self.select_goal))
        print("goal selection strategies: " + str(self.selection_strategies))
        print("goal selection strategie weights: " + str(self.strat_weights))
        print("goal reward function: " + str(self.compute_reward))
        print()

    def sample_obs_goal(self, obs = None):
        obs_space = self.env.observation_space

        # first trivial goal selection (uniform/default in obs space)
        return obs_space.sample()

    def multi_strat_goal_selection(self, obs = None):
        strat = self.np_random.choice(self.selection_strategies, p=self.strat_weights)

        return strat(obs)

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

    def select_goal_for_coverage(self, obs):
        # this temporary goal selection stratefy only tries to cover the space with
        # seen and targeted goals, to be improved later
        num_cand = 10

        # first goal is random
        if len(self.targeted_goals) < 1:
            return self.sample_obs_goal(obs)

        # TODO optimize this

        # make matrix of previous (seen and targeted) goals
        prev = np.stack(self.seen_goals + self.targeted_goals)

        # make matrix of candidate points
        candidate_points = np.stack([self.env.observation_space.sample() for i in range(num_cand)])

        # TODO normalize so that each dimension matters as much, with obs space size
        # or actual values seen in data?

        # calc distance between each candidate and each previous goals
        dist = scipy.spatial.distance.cdist(prev,candidate_points)

        # select candidate that is most "alone". This could be the one who's
        # closest neighbor is the furthest away for now.
        min_per_cand = np.min(dist, 0)
        ind = np.argmax(min_per_cand)
        best_cand = candidate_points[ind]

        return best_cand
