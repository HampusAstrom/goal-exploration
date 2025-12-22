import numpy as np
import utils
import matplotlib.pyplot as plt
import torch as th
import re

from goal_wrapper import FiveXGoalSelection
import scipy.stats as stats

import wandb

a = [[1, 0],
     [0, 1],
     [-1, 0],
     [0, -1],
     ]
b = [0,
     np.pi/2,
     np.pi,
     np.pi*3/2,
     ]

for i in range(len(a)):
    c = np.arctan2(a[i][1], a[i][0])
    c = ((c + np.pi) % (2 * np.pi)) - np.pi
    b2 = ((b[i] + np.pi) % (2 * np.pi)) - np.pi
    d = b2-c
    print(f"(x, y) = {a[i]}, expect {b2}, got {c}. diff = {d}")


print()

theta = [np.pi/4, 0.3, 0.1, np.pi/20, np.pi/4, 0.3, 0.1, np.pi/20, np.pi/4, 0.3, 0.1, np.pi/20]
thdot = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8]
u = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


for i in range(len(theta)):
    cost = theta[i] ** 2 + 0.1 * thdot[i]**2 + 0.001 * (u[i]**2)
    print(cost)

print()

x4 = np.cos(np.pi/4)
y4 = np.sin(np.pi/4)
x16 = np.cos(np.pi/16)
y16 = np.sin(np.pi/16)

a = np.asarray([[0, 1, 8],
                [1, 0, 8],
                [x4, y4, 8],
                [0, 1, 2],
                [1, 0, 2],
                [x4, y4, 2],
                [0, 1, 0],
                [1, 0, 0],
                [x4, y4, 0],
                [0, 1, -8],
                [1, 0, -8],
                [x4, y4, -8],
                [0, 1, -2],
                [1, 0, -2],
                [x4, y4, -2],
                [0, 1, -1],
                [1, 0, -1],
                [x4, y4, -1],
                [x16, y16, 8],
                [-x16, -y16, 8],
                [x16, y16, -8],
                [-x16, -y16, -8],
                [x16, y16, 4],
                [-x16, -y16, 2],
                [x16, y16, -4],
                [-x16, -y16, -2],
                ])

b = a

res = []
for i in range(len(a)):
    for j in range(len(b)):
        distance = np.linalg.norm(a[i] - b[j], axis=-1)
        #r = np.exp(-distance)
        r = distance
        res.append(r)
        print(r)

print(f"Mean: {np.mean(res)}")
print(f"Std: {np.std(res)}")
print(f"Min: {np.min(res)}")
print(f"Max: {np.max(res)}")

print(np.sort(res))

a = np.linspace(-10, 10)
b = utils.symlog(a)
a2 = utils.symexp(b)
c = utils.symexp(a)
a3 = utils.symlog(c)
fig = plt.figure()
ax = fig.add_subplot(221)
plt.plot(a, b)
ax = fig.add_subplot(222)
plt.plot(a, a2)
ax = fig.add_subplot(223)
plt.plot(a, c)
ax = fig.add_subplot(224)
plt.plot(a, a3)
plt.savefig("test_symlog")


dists = np.asarray([[0, 1, 3.5], [1, 0, 0], [1, 4, 2], [4, 4, 5]]).transpose()
rewards = np.asarray([1, 14, 3])

idw = FiveXGoalSelection.inverse_distance_weighting(dists, rewards)
print(idw)

idw = FiveXGoalSelection.inverse_distance_weighting_capped(dists, rewards, 4)
print(idw)

def test_func(a = 0, b = 0, c = 0):
    print(f"a = {a}, b = {b}, c = {c}")

params_to_permute = {"a": [0, 1, 2], "b": [3, 4, 5]}

from itertools import product, combinations
for combs in product (*params_to_permute.values()):
    dct = {ele: cnt for ele, cnt in zip(params_to_permute, combs)}
    print(dct)
    test_func(**dct)

# plot regular mountain car vs "pathological"
x = np.linspace(-1.7, 0.6, 100)
s = np.sin(3*x)/3
#p = ((-x**3) +(4*x**2)-4)*0.01
a = np.sin(3*x)/3-(0.15*x)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, s, label="sin")
plt.plot(x, a, label="alternate1")
#plt.plot(x, p, label="patho")
plt.savefig("view_car_plots")

max_s = x[np.argmax(s[:50])]
max_a = x[np.argmax(a[:50])]
print(max_s)
print(max_a)

indices = np.array([0, 1, 2, 3, 4,])
weights = np.array([1, 2, 3, 4, 5,])

combs = combinations(indices, 2)

#print(list(combs))
for comb in combs:
    comb = np.array(comb)
    conf = np.zeros(weights.shape)
    conf[comb] = weights[comb]
    print(conf)

combs = utils.weight_combinations(weights, 1)

print(combs)

for i in range(10):
    print(np.random.rand(2))


visits = np.array([0, 1, 5, 100, 1000])
svisits = sum(visits)
visits = visits/svisits

inv = 1/visits
inv_log = 1/np.log(visits)
inv_log_outer = np.log(1/visits)

softmin = th.nn.Softmin(visits)
softmax = th.nn.Softmax(visits)

smin = np.exp(-visits)
smin_n = smin/np.sum(smin)

print(inv)
print(inv_log)
print(inv_log_outer)
print(softmin)
print(softmax)
print(smin)
print(smin_n)
print(sum(smin_n))

print()
print(visits)
print(np.nonzero(visits))
print(visits[np.nonzero(visits)])

for index in np.ndindex((3, 3, 3)):
    print(index)


goal_s_grid = ["-1.384230418980660460e+00 2.180002849231407747e-03 4.754471396926380766e-01 2.237909566357616786e-02",
               "2.759791843732988248e-01 3.708371332509376173e-02 5.886782922112983041e-01 5.612880647526283240e-02",
               "-1.464574096820969062e+00 -2.894351247314004560e-02 -1.291288186954694828e+00 6.974597311914418341e-02",
               "4.834638234737749229e-01 -4.464850729668175255e-02 5.568313547549853526e-01 6.658949681702572287e-02",
               "-3.027599216509555546e-01 -6.182399811981372645e-02 3.333216901387956455e-01 6.172672486599169867e-02",
               "-1.466523939162593937e+00 -5.869430561829282189e-02 -1.237553626343519175e+00 -5.462094793453999009e-02",
               "-9.537505424345977811e-01 -1.608960307656900901e-02 -8.640412175094489555e-01 5.815403610048969912e-02",
]

goal_s_reselect_uni = ["-9.931327258174077466e-01 -1.252784453573574375e-02 -5.856670527671072879e-01 5.355620495845828111e-02",
          "-2.530850578270960050e-01 -4.313844291783267004e-02 4.650104154530999567e-01 -2.459363730692144118e-02",
          "-1.539896932106806338e+00 -4.063923724066526022e-03 -4.775276553121546863e-01 5.673771212032457933e-02",
          "-6.099934326966320874e-01 6.867769651365909778e-02 1.575302803047322620e-01 6.956311391379914333e-02",
          "-1.245749681790917940e+00 -4.546972892483388401e-02 9.506043291668087036e-02 -4.396294764963312729e-02",
          "-9.620841867647362822e-01 -5.047060233911408733e-02 3.341041207351336251e-01 5.559634799879557832e-02",
          "-1.387884598686021986e+00 3.143841966448764069e-02 5.424672442033751096e-01 4.303501001801912640e-02",
          "-1.644507169841727512e+00 -5.557678758598907920e-02 -9.711321678631266163e-01 1.328505095972538907e-02",
          "-3.229299152275477702e-02 2.478162851713962278e-02 2.715426162345616623e-01 6.847895962704436679e-02",
          "3.474738533364627457e-01 -2.013840012376874600e-02 5.074851286660113647e-01 9.630694740842233453e-03",
          "-4.216018409549138202e-01 -5.553199639807198523e-02 -3.056076286886412752e-01 6.903712227804126300e-02",
          "-1.681011939828521040e+00 -5.469338937699175229e-02 3.324544530074153847e-01 -2.387273654175500054e-02",
]

# initial goals with uniform goal selection (and fix to make not achieved initially)
goal_s = ["1.000859364867210388e-01 -6.027298465523560356e-02 3.867791221982637140e-01 4.441811234582084683e-02",
         "-3.567164242267608643e-01 -6.188835626306874588e-02 8.638786406774479065e-02 -8.343962442869223839e-03",
         "2.186840925893311915e-01 3.909031674265861511e-02 5.025510049536114909e-01 4.807805486983104254e-02",
         "4.036507010459899902e-01 -2.635284300500891769e-02 4.332456421129117219e-01 4.988685483677871707e-02",
         "-1.675310033343146987e+00 -3.845339848187752374e-02 -7.091766595840454102e-01 6.335049306681753145e-02",
         "-1.682318209836133338e+00 -6.895197661776852893e-02 -1.501349687576293945e+00 6.598497762876540107e-02",
         "3.431907892227172852e-01 -6.054840929603261951e-02 4.813866391003344125e-01 4.722256343471784024e-02",
         "-1.368107296119665195e+00 -6.381403995597903500e-02 4.192325005368322532e-01 -2.411544136703014374e-02",
         "4.851350188255310059e-01 5.278362594338584235e-02 5.691619110130864412e-01 6.848904487142416786e-02",
         "6.976728828170575269e-02 1.137789897620677948e-02 5.478681436619009526e-01 4.153764695377905325e-02",
         "-1.262233287470971188e+00 6.893713772296905518e-02 3.422347407134647135e-01 6.992038228802269217e-02",
         "-2.328613400459289551e-01 -6.845805063460036477e-02 4.574497339351997205e-01 3.243153282494198875e-02",
         "-1.515524311241561728e+00 1.842409372329711914e-02 -1.953139940458958890e-02 2.094209445655107957e-02",
         "-1.624003951982066596e+00 3.147694468498229980e-02 -7.203941191350495821e-01 6.432670089985989548e-02",
         "-1.379536608497856198e+00 1.368649117648601532e-02 5.971083418981897317e-01 2.756301659965686851e-02",
         "-3.512058352245157611e-01 6.733031570911407471e-02 5.141400348947161580e-01 6.810739938402812776e-02",
         "-1.612800064847025538e+00 -6.011686981719507883e-02 3.673420306537060753e-01 -1.871296577155590057e-02",
         "4.060764610767364502e-01 -6.170860602230360953e-02 4.075449458043788886e-01 -2.453653429281940287e-02",
         "-1.216826165130417303e+00 5.182468518614768982e-02 1.123383998019379382e-01 5.309569917887485557e-02",
         "-1.499770298344794028e+00 5.868653208017349243e-02 4.719213130954383884e-01 6.673022348424476058e-02",
         "3.073493242263793945e-01 -6.352007403953656428e-02 3.792282575519855814e-01 2.553215860997985753e-02",
         "-4.295613946776546399e-01 2.633811533451080322e-02 2.904275836582750903e-02 4.700026005675993213e-02",
         "-1.559658149531099269e+00 -4.156855965897648941e-02 -7.912279649823711347e-01 -9.375439956784248352e-03",
         "-6.111235376026542543e-01 1.011323556303977966e-02 1.759062782001198366e-01 3.327365683279218112e-02",
         "-7.854248864094415250e-01 -1.352308190231993335e-03 -2.136532736395803656e-01 -5.127451731823384762e-04",
]

def str2goal(string):
    goals = []
    for number in string.split():
        goals.append(float(number))
    return np.array(goals)

goals = []
for goal in goal_s:
    goals.append(str2goal(goal))

print(np.array(goals))

def plot_goals(goals):
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))

    ax.fill_between([0.5, 0.63], [0, 0], [0.07, 0.07],
                    alpha=0.5, fc="salmon", ec="red")
    ax.fill_between([-1.73, -1.6], [-0.07, -0.07], [0, 0],
                    alpha=0.5, fc="gold", ec="goldenrod")

    for goal in goals:
        x = [goal[0], goal[2]]
        ylow = [goal[1], goal[1]]
        yhigh = [goal[3], goal[3]]
        ax.fill_between(x, ylow, yhigh, alpha=0.5,)

plot_goals(goals)
#plt.show()

alpha = 0.05
for n in range(25):
    print(stats.t.ppf(1-(alpha/2), df=n-1))

def regex_each(regex, s_lst, keep=True, op=re.search):
    match_lst = []
    for s in s_lst:
        if keep and op(regex, s):
            print(f"keeping {op(regex, s)}")
            match_lst.append(s)
        elif not keep and op(regex, s):
            print(f"inverse {op(regex, s)}")
            match_lst.append(s)
    if keep:
        return match_lst
    else:
        return [s for s in s_lst if s not in match_lst]

strs = ["test", "eval/0", "eval/0_", "eval/1_"]

print("Regex test")
new_tags = regex_each(r'(/\d)', strs)
print(*new_tags, sep='\n')
print()
part = regex_each("(/0_)", new_tags, False)
print(*part, sep='\n')
both = regex_each("(/0_|/1_)", new_tags, False)
print(*both, sep='\n')

print()

a = [1, 2, 3]
b = ["a", "b", "c"]

c = [[1, "a"], [2, "b"], [3, "c"]]
zipped = zip(c)
print(list(zipped))
zipped = zip(*c)
print(list(zipped))

a = np.array([[1, 2], [3, 4]])
print(np.argmax(a))
a[0,0] = 4
print(np.argmax(a))

print("testing meta_eval_reward_quick_and_no_v_explode")

rew = [-1000, -1000, -1000, -1000]
init_v = [-1000, -1000, -1000, -1000]
steps = [1, 2, 3, 4]
max_steps = [4, 4, 4, 4]

window = 2

values = list(zip(rew, init_v, steps, max_steps))
meta_reward = utils.meta_eval_reward_quick_and_no_v_explode(values, rew_window=window)
print(values)
print(meta_reward)
print()

init_v[2] = 100
values = list(zip(rew, init_v, steps, max_steps))
meta_reward = utils.meta_eval_reward_quick_and_no_v_explode(values, rew_window=window)
print(values)
print(meta_reward)
print()

init_v[2] = -100
rew[1] = -110
values = list(zip(rew, init_v, steps, max_steps))
meta_reward = utils.meta_eval_reward_quick_and_no_v_explode(values, rew_window=window)
print(values)
print(meta_reward)
print()

rew[2] = -100
values = list(zip(rew, init_v, steps, max_steps))
meta_reward = utils.meta_eval_reward_quick_and_no_v_explode(values, rew_window=window)
print(values)
print(meta_reward)
print()

init_v[0] = 100
values = list(zip(rew, init_v, steps, max_steps))
meta_reward = utils.meta_eval_reward_quick_and_no_v_explode(values, rew_window=window)
print(values)
print(meta_reward)
print()


print(utils.obj2shortstr(10))
print(utils.obj2shortstr(-10))
print(utils.obj2shortstr(9))
print(utils.obj2shortstr(-999))
print(utils.obj2shortstr(-1.056))
print(utils.obj2shortstr(-0.00005))
print(utils.obj2shortstr(0.01))
print(utils.obj2shortstr(0.001))
print(utils.obj2shortstr(0.1))
print(utils.obj2shortstr(0.9))
print(utils.obj2shortstr(-0.9))

a = np.array(range(9)).reshape(3,3)
print(a)

sl = np.s_[0::2]

b = a[1,sl]
print(b)

b = a[:,sl]
print(b)

success_rate = [[0,0],[0,0.5],[1,1]]
init_v = [[0,0],[0.8,0.1],[0.3,0.2]]
dfr    = [[0,0],[0.2,0.1],[0.3,0.2]]

print(list(zip(success_rate, init_v, dfr)))

values = list(zip(success_rate, init_v, dfr))

ret = utils.meta_eval_goals(values)
print(ret)

dct = {"double_dqn": True, "use_sde": True}
algo = "SAC"
utils.filter_algo_kwargs_by_algo(dct, algo)
print(dct)

#algo = SAC
#algo_kwargs_merged["policy_kwargs"]["double_dqn"] = True
