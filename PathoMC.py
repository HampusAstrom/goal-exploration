########################################################################################################################
########## This is the code for Pathological Mountain Car introduced in Fig 1 #########################################
########  ENVIRONMENT WITH SPARSE REWARDS  #############################################################################
######################Input: state, action.... Output: Reward, New state, done info  ###################################
import numpy as np

def PathoMcar(state,action):

    done=False

    # action: gradient info
    grad_info = action
    state_new= state + 0.1*grad_info
    state_new = np.clip(state_new, -4, 3.709)
    fun_val = (-state_new**3) +(4*state_new**2)-4

    if np.abs(fun_val-5.481)< 0.001:
        reward =10
        done=True
    elif np.abs(fun_val-124)< 0.001:
        reward=500
        done =True
    else:
        reward =0

    # Reward for one step
    Reward_aggre = reward - 0.001*(grad_info**2)

    return np.array(state_new), Reward_aggre[0], done
