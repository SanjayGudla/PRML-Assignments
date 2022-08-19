import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
import random
import os
from kmeans_n_c import gen_kmeans
from gmm_n_c import gen_gmm
from sklearn.metrics import DetCurveDisplay

####data extraction

k =10

def calculate_gamma_nk(pi,mu,sigma,x_n):
    x_n = np.array(x_n)
    mu=np.array(mu)
    xn_mu = x_n - mu
    xn_mu = np.reshape(xn_mu,(len(x_n),1))

    exp = xn_mu.T @ np.linalg.inv(sigma) @ xn_mu
    exp /= -2

    temp = pi * math.exp(exp)
    temp /= math.sqrt(np.linalg.det(sigma))

    return temp

def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        for line in f.readlines():
            x=line.split()
            x = list(map(float,x))
            data.append(x)
    return data

def extractData1(directory):
    data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        data1 = []
        for line in f.readlines():
            x=line.split()
            x = list(map(float,x))
            data1.append(x)
        data.append(data1)
    return data


directories_train = ["./Features/coast/train","./Features/forest/train","./Features/highway/train","./Features/mountain/train","./Features/opencountry/train"]
directories_dev = ["./Features/coast/dev","./Features/forest/dev","./Features/highway/dev","./Features/mountain/dev","./Features/opencountry/dev"]
X_train_coast = extractData(directories_train[0])
X_train_forest = extractData(directories_train[1])
X_train_highway = extractData(directories_train[2])
X_train_mountain = extractData(directories_train[3])
X_train_opencountry = extractData(directories_train[4])
X_dev_coast = extractData1(directories_dev[0])
X_dev_forest = extractData1(directories_dev[1])
X_dev_highway = extractData1(directories_dev[2])
X_dev_mountain = extractData1(directories_dev[3])
X_dev_opencountry = extractData1(directories_dev[4])


X_dev = X_dev_coast+X_dev_forest+X_dev_highway+X_dev_mountain+X_dev_opencountry
C_dev = [1]*len(X_dev_coast)+[2]*len(X_dev_forest)+[3]*len(X_dev_highway)+[4]*len(X_dev_mountain)+[5]*len(X_dev_opencountry)


randval1 = random.sample(range(len(X_train_coast)),k)
randval2 = random.sample(range(len(X_train_forest)),k)
randval3 = random.sample(range(len(X_train_highway)),k)
randval4 = random.sample(range(len(X_train_mountain)),k)
randval5 = random.sample(range(len(X_train_opencountry)),k)

km_obj1 = gen_kmeans(X_train_coast,23,k,5,0,randval1)
km_out1 = km_obj1.kmeans_fun()
mu1,sigma1,pi1,cluster1 = km_out1

km_obj2 = gen_kmeans(X_train_forest,23,k,5,0,randval2)
km_out2 = km_obj2.kmeans_fun()
mu2,sigma2,pi2,cluster2 = km_out2

km_obj3 = gen_kmeans(X_train_highway,23,k,5,0,randval3)
km_out3 = km_obj3.kmeans_fun()
mu3,sigma3,pi3,cluster3 = km_out3

km_obj4 = gen_kmeans(X_train_mountain,23,k,5,0,randval4)
km_out4 = km_obj4.kmeans_fun()
mu4,sigma4,pi4,cluster4 = km_out4

km_obj5 = gen_kmeans(X_train_opencountry,23,k,5,0,randval5)
km_out5 = km_obj5.kmeans_fun()
mu5,sigma5,pi5,cluster5 = km_out5


# def predict_class_kmeans(mu1,mu2,mu3,mu4,mu5,k,point):
#     distance_array1 = [math.dist(point, mu1[i]) for i in range(k)]
#     min1 = min(distance_array1)
#     distance_array2 = [math.dist(point, mu2[i]) for i in range(k)]
#     min2 = min(distance_array2)
#     distance_array3 = [math.dist(point, mu3[i]) for i in range(k)]
#     min3 = min(distance_array3)
#     distance_array4 = [math.dist(point, mu4[i]) for i in range(k)]
#     min4 = min(distance_array4)
#     distance_array5 = [math.dist(point, mu5[i]) for i in range(k)]
#     min5 = min(distance_array5)
#     minimums = [min1,min2,min3,min4,min5]
#     minindex = minimums.index(min(minimums))
#     return minindex+1
#
# C_dev_found_kmeans=[]
# for point in X_dev:
#     C_dev_found_kmeans.append(predict_class_kmeans(mu1,mu2,mu3,mu4,mu5,k,point))


gmm_obj1 = gen_gmm(X_train_coast,23,k,5,km_out1,0)
gmm_obj2 = gen_gmm(X_train_forest,23,k,5,km_out2,0)
gmm_obj3 = gen_gmm(X_train_highway,23,k,5,km_out3,0)
gmm_obj4 = gen_gmm(X_train_mountain,23,k,5,km_out4,0)
gmm_obj5 = gen_gmm(X_train_opencountry,23,k,5,km_out5,0)
mu1,sigma1,pi1 = gmm_obj1.gmm_fun()
mu2,sigma2,pi2 = gmm_obj2.gmm_fun()
mu3,sigma3,pi3 = gmm_obj3.gmm_fun()
mu4,sigma4,pi4 = gmm_obj4.gmm_fun()
mu5,sigma5,pi5 = gmm_obj5.gmm_fun()

scores_gmm = []
def predict_class_gmm(mu1,mu2,mu3,mu4,mu5,pi1,pi2,pi3,pi4,pi5,sigma1,sigma2,sigma3,sigma4,sigma5,k,point):
    scores = np.array([0,0,0,0,0])
    for j in range(36):
        distance_array1 = [calculate_gamma_nk(pi1[i],mu1[i],sigma1[i],point[j]) for i in range(k)]
        min1 = sum(distance_array1)
        distance_array2 = [calculate_gamma_nk(pi2[i],mu2[i],sigma2[i],point[j]) for i in range(k)]
        min2 = sum(distance_array2)
        distance_array3 = [calculate_gamma_nk(pi3[i],mu3[i],sigma3[i],point[j]) for i in range(k)]
        min3 = sum(distance_array3)
        distance_array4 = [calculate_gamma_nk(pi4[i],mu4[i],sigma4[i],point[j]) for i in range(k)]
        min4 = sum(distance_array4)
        distance_array5 = [calculate_gamma_nk(pi5[i],mu5[i],sigma5[i],point[j]) for i in range(k)]
        min5 = sum(distance_array5)
        scores = scores + np.array([np.log(min1),np.log(min2),np.log(min3),np.log(min4),np.log(min5)])
    scores_gmm.append(scores)
    maxindex = np.argmax(scores)
    return maxindex+1

C_dev_found_gmm=[]
count =0
for point in X_dev:
    count+=1
    print(count)
    C_dev_found_gmm.append(predict_class_gmm(mu1,mu2,mu3,mu4,mu5,pi1,pi2,pi3,pi4,pi5,sigma1,sigma2,sigma3,sigma4,sigma5,k,point))

count1 = 0
for i in range(np.size(C_dev)):
    if (C_dev_found_gmm[i] == C_dev[i]):
        count1 = count1 + 1
print("Accuracy for GMM:- " + str(count1 / np.size(C_dev) * 100))


def plot_ROC_curve(scores, n,s):
    scores = np.array(scores)
    scores_mod = scores.flatten()
    scores_mod = np.sort(scores_mod)
    tpr = np.array([])
    fpr = np.array([])
    fnr = np.array([])
    for threshold in scores_mod:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(n):
            ground_truth = C_dev[i]
            for j in range(5):
                if (scores[i][j] >= threshold):
                    if ground_truth == j + 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j + 1:
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        fpr = np.append(fpr, fp / (fp + tn))
    plt.plot(fpr,tpr,label=s)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    return (fpr,tpr)

(fpr1,tpr1)=plot_ROC_curve(scores_gmm, len(C_dev),"gmm_nondiag")
plt.show()

fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=1-tpr1,estimator_name="case 1-gmm_nondiag").plot(ax=ax_det)
plt.show()

def draw_confusion(res1):
    conf= np.array([[0 for j in range(5)] for i in range(5)])
    for i in range(np.size(res1)):
        conf[C_dev[i]-1][res1[i]-1] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')
draw_confusion(C_dev_found_gmm)
plt.show()
