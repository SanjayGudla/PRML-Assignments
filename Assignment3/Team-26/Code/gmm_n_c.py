import mpmath
import math
import numpy as np


class gen_gmm:
    def __init__(self,data,n_dim,k_clusters,c_classes,kmeans_out,D):
        self.n_dim = n_dim
        self.k_cluster = k_clusters
        self.c_classes = c_classes
        self.data = data
        self.data_len = len(data)
        self.kmeans_out = kmeans_out
        self.D =D

    def calculate_gamma_nk(self,pi,mu,sigma,x_n):
        xn_mu = x_n - mu
        xn_mu = np.reshape(xn_mu,(self.n_dim,1))
        exp = xn_mu.T @ np.linalg.inv(sigma) @ xn_mu
        exp /= -2
        temp = pi * math.exp(exp)
        temp /= math.sqrt(np.linalg.det(sigma))
        return temp

    def gmm_fun(self):
        mu_s,sigmas,pi_s,clusters = self.kmeans_out
        for _ in range(1):
            gamma_nk_matrix = [[self.calculate_gamma_nk(pi_s[k], mu_s[k], sigmas[k], self.data[n])
                                for k in range(self.k_cluster)] for n in range(self.data_len)]

            sums = [sum(l) for l in gamma_nk_matrix]
            gamma_nk_matrix = [[val/sums[i] for val in gamma_nk_matrix[i]] for i in range(self.data_len)]

            pi_s = self.estimate_pi_k(gamma_nk_matrix)
            mu_s = self.estimate_mu_k(gamma_nk_matrix)
            sigmas = self.estimate_sigma_k(gamma_nk_matrix,mu_s)

        return mu_s,sigmas,pi_s

    def estimate_pi_k(self,gnk):
        pi_k = []
        for i in range(self.k_cluster):
            sums = 0
            for j in range(self.data_len):
                sums += gnk[j][i]

            sums /= self.data_len
            pi_k.append(sums)

        return pi_k

    def estimate_mu_k(self, gamma_nk_matrix):
        mu_k = []
        for k in range(self.k_cluster):
            sums = [0] * (self.n_dim)

            for n in range(self.data_len):
                p = self.data[n]
                sums = [sums[i] + gamma_nk_matrix[n][k]*p[i] for i in range(self.n_dim)]

            n_k = 0
            for i in range(self.data_len):
                n_k = n_k + gamma_nk_matrix[i][k]
            sums = [sums[i]/n_k for i in range(self.n_dim)]

            mu_k.append(sums)

        return mu_k

    def estimate_sigma_k(self, gamma_nk_matrix, mu_s):
        sigma_k = []
        g = np.matrix(gamma_nk_matrix)

        for k in range(self.k_cluster):
            n_k = sum(g[:,k])
            sums = np.zeros((self.n_dim,self.n_dim))

            for n in range(self.data_len):
                p=np.array(self.data[n])
                q= np.array(mu_s[k])
                xn_mu = p-q
                xn_mu = np.reshape(xn_mu, (self.n_dim, 1))

                temp = xn_mu @ xn_mu.T

                sums += np.multiply(temp, gamma_nk_matrix[n][k]/n_k)

                if self.D == 1:
                    sums = np.diag(np.diag(sums))
            sigma_k.append(sums)
        return sigma_k
