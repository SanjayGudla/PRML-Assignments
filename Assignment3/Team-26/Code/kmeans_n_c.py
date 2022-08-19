import math
import random

import numpy as np



class gen_kmeans:
    def __init__(self,data,n_dim,k_clusters,c_classes,D,rand_val):
        self.n_dim = n_dim
        self.k_cluster = k_clusters
        self.c_classes = c_classes
        self.data = data
        self.data_len = len(data)
        self.D = D
        self.rand_val = rand_val

    def calc_mu_sigma(self,cluster_data):
        cluster_size = len(cluster_data)

        nd_mean = np.zeros(self.n_dim, dtype=float)

        for x in cluster_data:
            nd_mean = nd_mean + np.array(x)

        nd_mean /= cluster_size

        nd_sigma = np.zeros((self.n_dim,self.n_dim),dtype=float)

        for x in cluster_data:
            a = x - nd_mean
            a = np.reshape(a,(self.n_dim,1))
            nd_sigma += a @ a.T

        nd_sigma /= cluster_size
        if self.D==1:
            nd_sigma = np.diag(np.diag(nd_sigma))

        return nd_mean,nd_sigma

    def kmeans_fun(self):
        mu_s = [self.data[i] for i in self.rand_val]
        for _ in range(10):
            clusters = [[] for i in range(self.k_cluster)]
            for point in self.data:
                dist_arr = [math.dist(point,mu_s[i]) for i in range(self.k_cluster)]
                min_pos = dist_arr.index(min(dist_arr))-1
                clusters[min_pos].append(point)

            clusters_mu_sigma = [self.calc_mu_sigma(clusters[k]) for k in range(self.k_cluster)]
            mu_s = [clusters_mu_sigma[k][0] for k in range(self.k_cluster)]
            sigmas = [clusters_mu_sigma[k][1] for k in range(self.k_cluster)]
            pi_s = [len(clusters[i])/self.data_len for i in range(self.k_cluster)]

        return mu_s,sigmas,pi_s,clusters
