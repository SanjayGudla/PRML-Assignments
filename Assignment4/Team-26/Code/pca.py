# Importiong Libraries
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
################################################################################
class PCA:
    def __init__(self,feature_vectors,feature_dim,reduced_dim):
        self.feature_vectors = feature_vectors
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim

    def mean_normalize(self,):
        n = len(self.feature_vectors)
        features_mean = np.zeros(self.feature_dim, dtype=float)
        for feature_vector in self.feature_vectors:
            features_mean = features_mean + np.array(feature_vector)
        features_mean /= n
        normalised_features = []
        for i in range(n):
            normalised_features.append((self.feature_vectors)[i]-features_mean)
        return np.array(normalised_features)

    def calc_cov(self,features):
        features_sigma = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
        for x in features:
            a = x
            a = np.reshape(a,(self.feature_dim,1))
            features_sigma += a @ a.T
        features_sigma /= len(self.feature_vectors)
        return features_sigma

    def myfn(self,x):
      return -abs(x)

    def pca_func(self):
        normalised_features = np.array(self.mean_normalize())
        covariance_matrix = np.array(self.calc_cov(normalised_features))
        eigVal,eigVec = LA.eig(covariance_matrix)
        eigVec = [eigVec[:, i] for i in range(len(eigVal))]

        eig_key = list(zip(eigVal, eigVec))
        sorted_eig_pairs = sorted(eig_key, key=lambda x: abs(x[0]),
                                  reverse=True)  # sorting the pair based on abs vaue of eigen value

        sorted_eig_val = [sorted_eig_pairs[i][0] for i in range(self.feature_dim)]
        sorted_eig_vec = [np.array(sorted_eig_pairs[i][1]) for i in range(self.feature_dim)]  # sorted eigen vectors and eigen values

        self.basis = sorted_eig_vec[:self.reduced_dim]

        Q = self.basis

        reduced_features = []
        for feature_vector in self.feature_vectors:
            reduced_feature = []
            for eigenvector in Q:
                reduced_feature.append((np.dot(np.array(feature_vector),eigenvector.T).real))
            reduced_features.append(reduced_feature)
        return reduced_features

    def project_test(self,test):
        coeff = []

        for vector in self.basis:
            coeff.append((np.dot(test,vector)).real)

        return coeff

    def project_test_array(self,test_array):
        coeffs = []
        for test in test_array:
            coeffs.append(self.project_test(test))
        return coeffs

    def project_data(self,data):
        ans = []
        for classfeatures in data:
            ans.append(self.project_test_array(classfeatures))
        return ans
