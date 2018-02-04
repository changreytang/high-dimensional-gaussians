#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from numpy import random
from scipy.stats import norm

def normalize(pt):
    return pt/np.linalg.norm(pt)

def angle_between(pt1, pt2):
    pt1_u = normalize(pt1)
    pt2_u = normalize(pt2)
    return np.arccos(np.clip(np.dot(pt1_u, pt2_u), -1.0, 1.0))

def plot(data, title):
    mu, std = norm.fit(data)
    plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    cap = "N = %d,  d = %d,  mu = %.2f,  std = %.2f" % (N, d, mu, std)
    plt.title(title + "\n" + cap)
    plt.show()

def dist_from_origin(rand_pts):
    origin = np.zeros(d, int)
    data = list()
    for pt in rand_pts:
        data.append(dist.euclidean(origin, pt))
    plot(data, "Distribution of Distance from Origin")

def dist_from_other_pts(rand_pts):
    data = list()
    for i in range(0,N*100):
        pt1 = rand_pts[random.randint(N)]
        pt2 = rand_pts[random.randint(N)]
        data.append(dist.euclidean(pt1, pt2))
    plot(data, "Distribution of Distance Between Pairs")

def angle_with_other_pts(rand_pts):
    data = list()
    for i in range(0,N*100):
        pt1 = rand_pts[random.randint(N)]
        pt2 = rand_pts[random.randint(N)]
        data.append(angle_between(pt1, pt2))
    plot(data, "Distribution of Angle Between Pairs")

def num_points_in_unit_ball(rand_pts):
    origin = np.zeros(d, int)
    num_inside = 0
    for pt in rand_pts:
        if dist.euclidean(origin, pt) < 1:
            num_inside += 1
    print("There are %d/%d points inside the unit ball, roughly %f percent" % (num_inside, N, num_inside/float(N)))

def num_points_in_sqrt_d_ball(rand_pts):
    origin = np.zeros(d, int)
    sqrtd = math.sqrt(d)
    num_inside = 0
    for pt in rand_pts:
        if dist.euclidean(origin, pt) < sqrtd:
            num_inside += 1
    print("There are %d/%d points inside the ball of radius sqrt(d), roughly %f percent" % (num_inside, N, num_inside/float(N)))

def main():
    mean = np.zeros(d, int)
    cov = np.zeros((d,d), int)
    np.fill_diagonal(cov, 1)

    origin = np.zeros(d, int)
    rand_pts = list()

    for i in range(0,N):
        rand_pts.append(random.multivariate_normal(mean, cov))

    if type == "dist_origin":
        dist_from_origin(rand_pts)
    elif type == "dist_pair":
        dist_from_other_pts(rand_pts)
    elif type == "angle":
        angle_with_other_pts(rand_pts)
    elif type == "unit_ball":
        num_points_in_unit_ball(rand_pts)
    elif type == "sqrtd_ball":
        num_points_in_sqrt_d_ball(rand_pts)
    else:
        print("USAGE: ./gaussian.py N d {dist_origin, dist_pair, angle, unit_ball, sqrtd_ball}")

    exit(1)

if len(sys.argv) != 4:
    print("USAGE: ./gaussian.py N d {dist_origin, dist_pair, angle, unit_ball, sqrtd_ball}")
    exit(1)

N = int(sys.argv[1])
d = int(sys.argv[2])
type = sys.argv[3]

if __name__ == "__main__":
    main()
