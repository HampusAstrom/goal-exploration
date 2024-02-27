import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from itertools import product

# a x, y space [0,1] with example trajectories.

rng = np.random.default_rng()

def closest_node(node, nodes):
    dist = distance.cdist([node], nodes)
    closest_index = dist.argmin()
    return closest_index, dist[0][closest_index]

def step(z, zT):
    step = rng.uniform(low=-0.02, high=0.08)
    # we could add wrapping here
    if zT > z:
        z += step
    else:
        z-=step
    return z

def path(x0, y0, xT, yT, steps = 20):
    x = x0
    y = y0

    points = [[x, y]]
    ri = [0]
    re = [0]
    for i in range(steps):
        x = step(x, xT)
        y = step(y, yT)
        points.append([x, y])
        if abs(xT-x) < 0.1 and abs(yT-y) < 0.1:
            re.append(1)
        else:
            re.append(0)
        # dummy placeholder for interest
        if i == 12 or i == 15:
            ri.append(1)
        else:
            ri.append(0)

    print(re)
    return points, ri, re

def plot_linepoints(x, y, weight, name):
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(x, y, '-b', alpha=0.1)
    #ax1.plot(x, y, 'ob', alpha=1)
    ax1.scatter(x, y, s=50, c=weight, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    #fig1.tight_layout()
    plt.savefig(name)

# this stupidly? only cares about closest node, can snuff out other close values
def close_to_r_pf(dist, r):
    if dist < 0.04*r:
        return True, 0.04*r
    return False, None

def close_to_r_gf(dist, r):
    if dist < 0.1*r:
        return True, 0.04*r
    return False, None

def goldilocks_gf(dist, r):
    if dist < 0.12 and dist > 0.04:
        return True, None
    return False, None

def goldilocks_pf(dist, r):
    if dist < 0.08 and dist > 0.06:
        return True, None
    return False, None

def goldilocks(points, r, pf, gf, bf):
    xt = np.linspace(0, 1, num=100)
    yt = np.linspace(0, 1, num=100)
    bad = []
    good = []
    perfect = []
    if pf is None:
        pf = goldilocks_pf
    if gf is None:
        gf = goldilocks_gf
    for x, y in product(xt, yt):
        ind, dist = closest_node([x, y], points)
        usep, valp = pf(dist, r[ind])
        useg, valg = gf(dist, r[ind])
        if bf is not None:
            useb, valb = bf(dist, r[ind])
        if usep:
            perfect.append([x, y])
        elif useg:
            good.append([x, y])
        elif bf is not None and useb:
            bad.append([x, y])
    return perfect, good, bad

def plot_goldilocks(points, r, name, pf=None, gf=None, bf=None):
    perfect, good, bad = goldilocks(points, r, pf, gf, bf)
    x, y = zip(*points)
    size = 20
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    #ax1.plot(x, y, 'ob', alpha=1)
    if len(bad) > 0:
        xb, yb = zip(*bad)
        ax1.scatter(xb, yb, s=size, c='r', alpha=1)    
    if len(good) > 0:
        xg, yg = zip(*good)
        ax1.scatter(xg, yg, s=size, c='y', alpha=1)
    if len(perfect) > 0:
        xp, yp = zip(*perfect)
        ax1.scatter(xp, yp, s=size, c='g', alpha=1)
    ax1.plot(x, y, '-bo', alpha=0.1)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    #fig1.tight_layout()
    plt.savefig(name)


def plot(points, ri, re):
    # lets start by just plotting the points
    x, y = zip(*points)
    #plot_linepoints(x, y, ri, "ri")
    #plot_linepoints(x, y, re, "re")
    plot_goldilocks(points, ri, "goldilocks")
    plot_goldilocks(points, ri, "ri", pf=close_to_r_pf, gf=close_to_r_gf)
    plot_goldilocks(points, re, "re", pf=close_to_r_pf, gf=close_to_r_gf)


points, ri, re = path(0.5, 0.2, 0.9, 0.9)

plot(points, ri, re)


#a = [[0,1], [1, 1], [2, 1]]

#print(closest_node([1.6, 1], a))

