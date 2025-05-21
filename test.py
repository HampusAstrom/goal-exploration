import numpy as np
import utils
import matplotlib.pyplot as plt
import torch as th

from goal_wrapper import FiveXGoalSelection

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