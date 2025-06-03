import gymnasium as gym
import numpy as np

def predict(obs):
    oracle = [ 0,  3,  0,  3, # third value 0 or 3?
               0, -1,  2, -1,
               3,  1,  0, -1,
              -1,  2,  1, -1]
    return oracle[obs]


env = gym.make("FrozenLake-v1", is_slippery=True)

n_exps = 1000000
trunk_limit = 100
rewards = []
trunks = 0
terms = 0
failures = 0
successes = 0

for i in range(n_exps):
    totrew = 0
    obs, _ = env.reset()
    for j in range(trunk_limit):
        action = predict(obs)
        obs, reward, term, trunk, info = env.step(action)
        totrew += reward
        if trunk:
            trunks += 1
        if term:
            terms += 1
            if reward < 1:
                failures += 1
            else:
                successes += 1
            break
        #env.render("human")
    rewards.append(totrew)

#print(rewards)
print(f"Mean reward: {np.mean(rewards)}")
print(f"Trunk %:     {trunks/n_exps}")
print(f"Term %:      {terms/n_exps}")
print(f"Failure %:   {failures/n_exps}")
print(f"Failure %:  {successes/n_exps}")

# Results 1000000 exps:
# Mean reward: 0.692747
# Trunk %:     0.049082
# Term %:      0.952619
# Failure %:   0.259872
# Failure %:  0.692747

# Apparently ideal should be 0.74
