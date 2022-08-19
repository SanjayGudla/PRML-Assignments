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
class LDA:
    def __init__(self,feature_vectors,feature_dim,reduced_dim):
        self.feature_vectors = feature_vectors
        self.feature_dim = feature_dim
        self.reduced_dim = reduced_dim

    def calculate_mean(self,class_features):
        n = len(class_features)
        class_mean = np.zeros(self.feature_dim, dtype=float)
        for feature_vector in class_features :
            class_mean = class_mean + feature_vector
        class_mean /= n
        return class_mean

    def calculate_total_mean(self,):
        total_mean = np.zeros(self.feature_dim, dtype=float)
        n=0
        for class_features in self.feature_vectors:
            for feature_vector in class_features :
                total_mean = total_mean + feature_vector
                n=n+1
        total_mean /= n
        return total_mean

    def calculate_sw(self,features):
        num_classes = len(self.feature_vectors)
        sw = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
        for class_features in self.feature_vectors:
            mk = self.calculate_mean(class_features)
            sk = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
            for xn in class_features:
                xn = np.reshape(xn,(self.feature_dim,1))
                mk = np.reshape(mk,(self.feature_dim,1))
                sk = sk + (xn-mk)@(xn-mk).T
            sw = sw+sk
        return sw

    def calculate_st(self,features):
        num_classes = len(self.feature_vectors)
        st = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
        mk = self.calculate_total_mean()
        for class_features in self.feature_vectors:
            sk = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
            for xn in class_features:
                xn = np.reshape(xn,(self.feature_dim,1))
                mk = np.reshape(mk,(self.feature_dim,1))
                sk = sk + (xn-mk)@(xn-mk).T
            st = st+sk
        return st

    def lda_func(self):
        sw = self.calculate_sw(self.feature_vectors)
        st = self.calculate_st(self.feature_vectors)
        x = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
        y = np.zeros((self.feature_dim,self.feature_dim),dtype=float)
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                x[i][j]=sw[i][j]
                y[i][j]=st[i][j]
        y=y-x
        eigVal,eigVec = LA.eig(LA.inv(x)@y)
        eigVec = [eigVec[:, i] for i in range(len(eigVal))]
        eig_key = list(zip(eigVal, eigVec))
        sorted_eig_pairs = sorted(eig_key, key=lambda x: abs(x[0]),
                                  reverse=True)  # sorting the pair based on abs vaue of eigen value
        sorted_eig_val = [sorted_eig_pairs[i][0] for i in range(self.feature_dim)]
        sorted_eig_vec = [np.array(sorted_eig_pairs[i][1]) for i in range(self.feature_dim)]  # sorted eigen vectors and eigen values
        self.basis = sorted_eig_vec[:self.reduced_dim]
        Q = self.basis
        reduced_features = []
        for class_features in self.feature_vectors:
            for feature_vector in class_features:
                reduced_feature = []
                for eigenvector in Q:
                    reduced_feature.append((np.dot(np.array(feature_vector).T,eigenvector)).real)
                reduced_features.append(reduced_feature)
        return reduced_features

    def project_test(self,test):
        coeff = []
        for vector in self.basis:
            coeff.append((np.dot(test,vector)).real)
        return coeff

    def project_test_array(self,test_array):
        reduced_test_array = []
        for  test in test_array:
            reduced_test = self.project_test(test)
            reduced_test_array.append(reduced_test)
        return reduced_test_array
