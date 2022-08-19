import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
import os

####data extraction

def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            for line in lines[1:]:
                x=line.split()
                x = list(map(float, x))
                m.append(x)
            data.append(m)
    return np.array(data)

def dtw(s,t):
        s = np.array(s)
        t = np.array(t)
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                temp = s[i - 1] - t[j - 1]
                cost = np.dot(temp.T,temp)
                cost = math.sqrt(cost)
                last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n][m]

directories_train = ["./3/train","./5/train","./6/train","./8/train","./z/train"]
directories_dev = ["./3/dev","./5/dev","./6/dev","./8/dev","./z/dev"]
X_train3 = extractData(directories_train[0])
X_train5 = extractData(directories_train[1])
X_train6 = extractData(directories_train[2])
X_train8 = extractData(directories_train[3])
X_trainz = extractData(directories_train[4])

X_dev3 = extractData(directories_dev[0])
X_dev5 = extractData(directories_dev[1])
X_dev6 = extractData(directories_dev[2])
X_dev8 = extractData(directories_dev[3])
X_devz = extractData(directories_dev[4])

scores1 = []
scores2 = []
scores3 = []
scores4 = []
C_dev_found = []
def find_output(X_dev,original_class):
    total_count = 0
    matched_count = 0
    for dev_point in X_dev:
        dist_arr1,dist_arr2,dist_arr3,dist_arr4,dist_arr5 =[],[],[],[],[]
        for train_point in X_train3:
            dist_arr1.append(dtw(train_point,dev_point))
        for train_point in X_train5:
            dist_arr2.append(dtw(train_point,dev_point))
        for train_point in X_train6:
            dist_arr3.append(dtw(train_point,dev_point))
        for train_point in X_train8:
            dist_arr4.append(dtw(train_point,dev_point))
        for train_point in X_trainz:
            dist_arr5.append(dtw(train_point,dev_point))

        k = 2
        dist_arr1,dist_arr2,dist_arr3,dist_arr4,dist_arr5 = sorted(dist_arr1),sorted(dist_arr2),sorted(dist_arr3),sorted(dist_arr4),sorted(dist_arr5)
        scores_i = [k/sum(dist_arr1[:k]),k/sum(dist_arr2[:k]),k/sum(dist_arr3[:k]),k/sum(dist_arr4[:k]),k/sum(dist_arr5[:k])]
        output_class = scores_i.index(max(scores_i))
        scores1.append(scores_i)
        k = 5
        dist_arr1,dist_arr2,dist_arr3,dist_arr4,dist_arr5 = sorted(dist_arr1),sorted(dist_arr2),sorted(dist_arr3),sorted(dist_arr4),sorted(dist_arr5)
        scores_i = [k/sum(dist_arr1[:k]),k/sum(dist_arr2[:k]),k/sum(dist_arr3[:k]),k/sum(dist_arr4[:k]),k/sum(dist_arr5[:k])]
        output_class = scores_i.index(max(scores_i))
        scores2.append(scores_i)
        k = 8
        dist_arr1,dist_arr2,dist_arr3,dist_arr4,dist_arr5 = sorted(dist_arr1),sorted(dist_arr2),sorted(dist_arr3),sorted(dist_arr4),sorted(dist_arr5)
        scores_i = [k/sum(dist_arr1[:k]),k/sum(dist_arr2[:k]),k/sum(dist_arr3[:k]),k/sum(dist_arr4[:k]),k/sum(dist_arr5[:k])]
        output_class = scores_i.index(max(scores_i))
        scores3.append(scores_i)
        k = 10
        dist_arr1,dist_arr2,dist_arr3,dist_arr4,dist_arr5 = sorted(dist_arr1),sorted(dist_arr2),sorted(dist_arr3),sorted(dist_arr4),sorted(dist_arr5)
        scores_i = [k/sum(dist_arr1[:k]),k/sum(dist_arr2[:k]),k/sum(dist_arr3[:k]),k/sum(dist_arr4[:k]),k/sum(dist_arr5[:k])]
        output_class = scores_i.index(max(scores_i))
        scores4.append(scores_i)

        C_dev_found.append(output_class)
        if output_class == original_class:
            matched_count = matched_count + 1
        total_count = total_count+1
    print(matched_count,total_count)
    return (matched_count,total_count)

(a,b)=find_output(X_dev3,0)
(c,d)=find_output(X_dev5,1)
(e,f)=find_output(X_dev6,2)
(g,h)=find_output(X_dev8,3)
(i,j)=find_output(X_devz,4)

print("Accuracy is :-" + str((a+c+e+g+i)/(b+d+f+h+j) * 100)+"%.")
C_dev = [0]*len(X_dev3)+[1]*len(X_dev5)+[2]*len(X_dev6)+[3]*len(X_dev8)+[4]*len(X_devz)
n = len(C_dev)
scores1 = np.array(scores1)
scores2 = np.array(scores2)
scores3 = np.array(scores3)
scores4 = np.array(scores4)

def plot_ROC_curve(scores, n):
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
                    if ground_truth == j :
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j :
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        fpr = np.append(fpr, fp / (fp + tn))
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    return (fpr,tpr)

(fpr1,tpr1)=plot_ROC_curve(scores1,n)
(fpr2,tpr2)=plot_ROC_curve(scores2,n)
(fpr3,tpr3)=plot_ROC_curve(scores3,n)
(fpr4,tpr4)=plot_ROC_curve(scores4,n)
plt.show()

def draw_confusion(res1):
    conf= np.array([[0 for j in range(5)] for i in range(5)])
    for i in range(np.size(res1)):
        conf[C_dev[i]][res1[i]] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')
draw_confusion(C_dev_found)
plt.show()

fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=1-tpr1,estimator_name="DTW2").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr2,fnr=1-tpr2,estimator_name="DTW5").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr3,fnr=1-tpr3,estimator_name="DTW8").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr4,fnr=1-tpr4,estimator_name="DTW10").plot(ax=ax_det)
plt.show()
