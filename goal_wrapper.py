import gymnasium as gym
import numpy as np
import scipy
from itertools import product

from gymnasium import error, logger
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.utils import flatten, flatdim
from gymnasium import spaces
import utils

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
            goal_weight: float=0.5,
            reward_func = None, # TODO make type hint with Callable
            goal_selection_strategies = None, # TODO make type hint with Callable
            goal_sel_strat_weight = None,
            # env: gym.Env[ObsType, ActType],
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            goal_weight=goal_weight,
        )
        gym.Wrapper.__init__(self, env)

        assert goal_weight >=0 and goal_weight <= 1
        self.goal_weight = goal_weight
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
        self.targeted_goals = []

    def step(
            self,
            action
            ):
            #) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]: # TODO remove or fix imports of types and such
        obs, reward, terminated, truncated, info = self.env.step(action)

        # get goal conditioned reward
        goal_reward = self.compute_reward(obs, self.goal, info)
        gw = self.goal_weight
        # output weighted average of rewards
        totreward = gw*goal_reward + (1-gw)*reward

        # save separate rewards in info
        assert info.get("extrinsic_reward") == None
        assert info.get("goal_reward") == None
        info["extrinsic_reward"] = reward
        info["goal_reward"] = goal_reward
        # TODO add intrinsic reaward here, unless already added by lower wrapper?

        #self.seen_goals.append(obs)

        return self._get_obs(obs), totreward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, reset_info = self.env.reset(**kwargs)
        self.goal = self.select_goal(obs) # goal selection can require obs info

        self.targeted_goals.append(self.goal)

        return self._get_obs(obs), reset_info

    def _get_obs(self, obs):
        # current version assumes goals in obs space, expand this when that is not
        # always the case anymore
        # maybe make a check on class creation and skipp "achieved_goal" when
        # same as regular obs, for efficent memory usage
        return {"observation": obs, "achieved_goal": obs, "desired_goal": self.goal}

    def set_goal_strategies(self, goal_selection_strategies, goal_sel_strat_weight=None):
        # if list of selection strategies but not weights, uniform is assumed
        self.selection_strategies = goal_selection_strategies
        if goal_sel_strat_weight is None:
            self.strat_weights = [1]*len(goal_selection_strategies)
        else:
            self.strat_weights = goal_sel_strat_weight
        self.select_goal = self.multi_strat_goal_selection

    def link_buffer(self, buffer):
        self.replay_buffer = buffer

    def print_setup(self):
        print()
        print("Goal wrapper setup")
        print("goal weight: " + str(self.goal_weight))
        print("goal selection method: " + str(self.select_goal))
        print("goal selection strategies: " + str(self.selection_strategies))
        print("goal selection strategy weights: " + str(self.strat_weights))
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
        distance = np.linalg.norm(flat_achi - flat_goal, axis=-1)
        # for now returning dense reward, hard to extimate generic cutoff value
        return np.exp(-distance)

class FiveXGoalSelection():
    def __init__(self,
                 env,
                 replay_buffer,
                 targeted_goals_list,
                 num_candidates = 10,
                 component_weights = [1, 1, 5, 1, 1],
                 explain_dist = 0.1,
                 exploit_dist = 0.1,) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.targeted_goals = targeted_goals_list
        self.components_for_candidates = []
        self.num_candidates = num_candidates
        self.component_weights = component_weights
        self.explain_dist = explain_dist
        self.exploit_dist = exploit_dist

    def norm_each_dim(self, array):
        # normalizes each sample in array to be on the range of 0-1 in each dimension
        # does not work for inf dims...
        # TODO replace with norm that matches sampling strategy for
        # bounded/unbounded in https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/box.py
        low = self.env.unwrapped.observation_space.low
        high = self.env.unwrapped.observation_space.high
        ret = (array - low)/(high - low)
        return ret

    def grid_of_points(self, points_per_dim):
        # TODO this method should probably be in utils and take an env instead
        o_space = self.env.unwrapped.observation_space
        grid_per_dim = np.linspace(o_space.low,
                                 o_space.high,
                                 points_per_dim,
                                 )
        ret = list(product(*grid_per_dim.T))
        return np.array(ret)

    @staticmethod
    def goldilocks(dists):
        # assumes normalized dimensions
        # TODO make a tunable or adapt on their own based on experience
        # TODO determine if this function is too "flat"? maybe replace with
        # gaussian
        a = 0.1 # sets distance for maximum, with a max value of 1
        beta = 1/a**2
        alpha = 2/a
        return np.maximum(alpha*dists - beta*dists**2, 0)

    @staticmethod
    def inverse_distance_weighting_capped(dists, rewards, range):
        mean_reward = np.mean(rewards)
        zones = np.maximum(1-dists/range, 0)
        zones_sum =  np.sum(zones, 0)
        contrib_per_seen = zones*rewards[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = np.sum(contrib_per_seen, 0)/zones_sum
        # replace nan:s (from when no point in area) with mean of all points
        contrib[np.isnan(contrib)] = mean_reward
        return contrib

    @staticmethod
    def inverse_distance_weighting(dists, rewards):
        with np.errstate(divide='ignore'):
            contrib_per_seen = rewards[:, None]/dists
            print(contrib_per_seen)
            contrib = np.sum(contrib_per_seen, 0)/np.sum(dists, 0)
        # replace nan:s (from when exact point is seen) with that point
        to_replace = np.logical_or(np.isnan(contrib_per_seen),
                                   np.isinf(contrib_per_seen))
        contrib[np.any(to_replace, 0)] = rewards[np.where(np.any(to_replace, axis=0))]
        return contrib

    def select_goal_for_coverage(self,
                                 obs,
                                 fixed_candidates = None,
                                 map_not_choose = False):
        # TODO consider and possibly implement way to change component weights
        # over time with a scheduler. We might want more exploit later, for
        # instance, but maybe also more experiment/expand early? unclear

        # TODO determine if there is a nice way to make all components somewhat
        # balanced, on average or all the time (before weights)

        # first goal is random
        if len(self.targeted_goals) < 1:
            return self.env.unwrapped.observation_space.sample()

        # TODO doesn't handle multiple environments, fix
        rb = self.replay_buffer
        upper_bound = rb.buffer_size if rb.full else rb.pos # all before have data
        observations = self.replay_buffer.observations["observation"][:upper_bound,0]
        # TODO reward parts being storead in dict is inefficient, can we fix it
        # without breaking the api?
        ext_rewards = np.array([l["extrinsic_reward"] for l in \
                                self.replay_buffer.infos[:upper_bound,0]])
        # normalize with symlog like Dreamer v3
        ext_rewards = utils.symlog(ext_rewards)

        # TODO make sure that we have policy that saves intrinsic rewards here
        int_rewards = np.array([l.get("intrinsic_reward", 0) for l in \
                                self.replay_buffer.infos[:upper_bound,0]])
        int_rewards = utils.symlog(int_rewards)

        # TODO determine if we need to flatten stuff here too for the general case
        # TODO optimize this
        # TODO mock data and make testcase to verify each component and result

        # make matrix of candidate points
        if fixed_candidates is not None:
            candidate_points = fixed_candidates
        else:
            candidate_points = np.stack([self.env.unwrapped.observation_space.sample() \
                                for i in range(self.num_candidates)])

        # normalize so that each dimension matters as much with obs space size
        # TODO or actual values seen in data?
        # or with symlog, but that would need to be on distances not coordinates
        # TODO apparently there is a env.normalize_obs
        # I should check if I can use that instead, it is a running average
        n_candidate_points = self.norm_each_dim(candidate_points)
        seen_goals = self.norm_each_dim(observations)
        targeted_goals = self.norm_each_dim(self.targeted_goals)

        # calc distance between each candidate and each previous goals
        seen_dists = scipy.spatial.distance.cdist(seen_goals,n_candidate_points)
        targeted_dists = scipy.spatial.distance.cdist(targeted_goals,n_candidate_points)
        all_dists = np.concatenate((seen_dists, targeted_dists))

        # select components to care about if not all at all times

        # get experiment component
        # select candidate that is most "alone". This could be the one who's
        # closest neighbor is the furthest away for now.
        min_dist_to_any = np.min(all_dists, 0)

        # get expand (goldilocks) component
        min_dist_to_seen = np.min(seen_dists, 0)
        goldilocks = self.goldilocks(min_dist_to_seen)

        # get exclude component
        # TODO discount exclusion over time
        # for now we make it a mean (by dividing by number of targeted goals)
        # the size of each exclusion on average shrinks with new data, that
        # should be enough
        targeted2seen_dists = scipy.spatial.distance.cdist(seen_goals,targeted_goals)
        exclusion_sizes = np.min(targeted2seen_dists, 0)
        exclusion_per_target = - np.maximum(1-targeted_dists/exclusion_sizes[:, None], 0)
        exclusion_contrib = np.sum(exclusion_per_target, 0)

        # get explain component
        explain_contrib = self.inverse_distance_weighting_capped(seen_dists,
                                                                 int_rewards,
                                                                 self.explain_dist)

        # get exploit component
        exploit_contrib = self.inverse_distance_weighting_capped(seen_dists,
                                                                 ext_rewards,
                                                                 self.exploit_dist)

        # TODO make it easy to choose capped and non-capped exploit/explain

        components_for_candidates = np.array([self.component_weights[0] * min_dist_to_any,
                                     self.component_weights[1] * goldilocks,
                                     self.component_weights[2] * exclusion_contrib,
                                     self.component_weights[3] * explain_contrib,
                                     self.component_weights[4] * exploit_contrib,])

        # compine components and select best candidate
        # TODO make an option to only select some components sometimes, possibly
        # by selecting each goal with some relative weight based on component
        # weight in a way where many are selected, not just one?
        # also maybe always exclude?
        total_goal_val = np.sum(components_for_candidates, 0)

        if not map_not_choose:
            self.components_for_candidates.append(components_for_candidates)
        else:
            return components_for_candidates

        for part in components_for_candidates:
            print(np.array2string(np.array(part), sign=' ', precision=3))
        print(total_goal_val)
        print(np.max(total_goal_val))
        print()
        # select cantidate with highest weight and return its goal
        ind = np.argmax(total_goal_val)
        best_cand = candidate_points[ind]

        return best_cand
