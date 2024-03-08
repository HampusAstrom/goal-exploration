########################################################################################################################
########## This is the implementation of Mario 1D problem: based on
######## THE PROBLEM WITH DDPG: UNDERSTANDING FAILURES IN DETERMINISTIC ENVIRONMENTS WITH SPARSE REWARDS
##############################Input: state, action.... Output: Reward, New state  ######################################

import numpy as np

def Mario(state,action):
    done=False
    grad_info = action
    state_new= np.minimum(1.0,np.maximum(0,state+action))

    if (state+action)<0:
        reward = 1
    else:
        reward = 0

    # Used the  additional part given below
    if state_new ==0.0:
       done = True

    # Reward for one step
    Reward_aggre = reward
    return np.array(state_new), Reward_aggre, done
