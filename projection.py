#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from numpy import random
from scipy.stats import norm
from sklearn import random_projection
import decimal

def calculate_distance_between_all_pairs(pts):
    dist_between_pts = list()
    for i in range(0,len(pts)):
        for j in range(i+1,len(pts)):
            dist_between_pts.append(dist.euclidean(pts[i],pts[j]))
    return dist_between_pts

def project_data(pts, d, k):
    #mean = np.zeros(d, int)
    #cov = np.zeros((d,d), int)
    #np.fill_diagonal(cov, 1)
    #ulist = list()
    #for i in range(0, k):
    #    ulist.append(random.multivariate_normal(mean,cov))
    #projected_data = list()
    #for pt in pts:
    #    projected_pt = list()
    #    for u in ulist:
    #        projected_pt.append(np.dot(pt,u))
        #print(str(np.linalg.norm(projected_pt)) + ":" + str(math.sqrt(k)*np.linalg.norm(pt)))
    #    projected_data.append(np.array(projected_pt))
    transformer = random_projection.GaussianRandomProjection(k)
    projected_data = transformer.fit_transform(pts)
    #print(projected_data.shape)
    return projected_data

def find_max_difference(orig_dist, proj_dist, k):
    sqrtk = math.sqrt(k)
    sqrtk_orig_dist = [d*sqrtk for d in orig_dist]
    max_diff = 0
    for i in range(0,len(orig_dist)):
        cur_diff = abs(orig_dist[i] - proj_dist[i])
        if (cur_diff > max_diff):
            max_diff = cur_diff
    print("The maximum difference is %d which is roughly %f percent  of k=%d" % (max_diff, (max_diff/k)*100, k))


def main():
    d = 900
    r = 30
    origin = np.zeros(d, int)
    mean = np.zeros(d, int)
    cov = np.zeros((d,d), int)
    np.fill_diagonal(cov, 1)

    origin = np.zeros(d, int)
    rand_pts = list()

    while len(rand_pts) != 20:
        rand_pt = random.multivariate_normal(mean,cov)
        if dist.euclidean(origin, rand_pt) < r:
            rand_pts.append(rand_pt)

    #print(rand_pts)
    original_dists = calculate_distance_between_all_pairs(rand_pts)
    #print(original_dists)
    projected_100 = project_data(rand_pts, d, 100)
    #print(projected_100)
    projected_50 = project_data(rand_pts, d, 50)
    projected_10 = project_data(rand_pts, d, 10)
    projected_5 = project_data(rand_pts, d, 5)
    projected_4 = project_data(rand_pts, d, 4)
    projected_3 = project_data(rand_pts, d, 3)
    projected_2 = project_data(rand_pts, d, 2)
    projected_1 = project_data(rand_pts, d, 1)
    #print(projected_100)
    dists_100 = calculate_distance_between_all_pairs(projected_100)
    #print(dists_100)
    dists_50 = calculate_distance_between_all_pairs(projected_50)
    dists_10 = calculate_distance_between_all_pairs(projected_10)
    dists_5 = calculate_distance_between_all_pairs(projected_5)
    dists_4 = calculate_distance_between_all_pairs(projected_4)
    dists_3 = calculate_distance_between_all_pairs(projected_3)
    dists_2 = calculate_distance_between_all_pairs(projected_2)
    dists_1 = calculate_distance_between_all_pairs(projected_1)
    #print(dists_100)
    find_max_difference(original_dists, dists_100, 100)
    find_max_difference(original_dists, dists_50, 50)
    find_max_difference(original_dists, dists_10, 10)
    find_max_difference(original_dists, dists_5, 5)
    find_max_difference(original_dists, dists_4, 4)
    find_max_difference(original_dists, dists_3, 3)
    find_max_difference(original_dists, dists_2, 2)
    find_max_difference(original_dists, dists_1, 1)

if __name__ == "__main__":
    main()
