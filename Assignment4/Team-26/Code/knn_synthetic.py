from knn_module import KNN
from pca import PCA
import numpy as np
import matplotlib.pyplot as plt
from plots import plots
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
f1 = "./Synthetic_Data/26/train.txt"
f2 = "./Synthetic_Data/26/dev.txt"

def fetch_data(f):
    file = open(f, "r")
    C = []
    X = []
    while True:
        line = file.readline()
        if not line:
            break
        x, y, c = line.strip().split(',')
        C.append(int(c))
        X.append((np.longdouble(x), np.longdouble(y)))
    return (C, X)

c_train, x_train = fetch_data(f1)
c_test, x_test = fetch_data(f2)

reduced_train = x_train
reduced_dev = x_test

# pca_obj = PCA(x_train,2,1)
# reduced_train = pca_obj.pca_func()
#
# reduced_dev = []
#
# for vector in x_test:
#     reduced_dev.append(pca_obj.project_test(vector))

fpr_arr = []
tpr_arr = []
accuracy = []
k_arr = [80,100,120,140,160,180,260,300,400,500]
for  k in k_arr :
    knn_obj = KNN(reduced_train,reduced_dev,c_train,c_test,2,k)
    knn_obj.compute_distance()
    prediction,scores = knn_obj.classify()
    count = 0
    for i in range(len(c_test)):
        if c_test[i] == prediction[i]:
            count += 1
    accuracy.append(count / len(c_test) * 100)
    print(count / len(c_test) * 100)
    (fpr,tpr) = plots.plot_ROC_curve(np.array(scores),len(c_test),2,c_test)
    fpr_arr.append(fpr)
    tpr_arr.append(tpr)


for it in range(len(fpr_arr)):
    plt.plot(fpr_arr[it],tpr_arr[it],label=("k = "+str(k_arr[it])))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
plt.legend()
plt.show()


fig, ax_det = plt.subplots(1,1)
for it in range(len(fpr_arr)):
    d1=DetCurveDisplay(fpr=fpr_arr[it],fnr=1-tpr_arr[it],estimator_name=("k = "+str(k_arr[it]))).plot(ax=ax_det)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET  Curve")
plt.show()

plt.plot(k_arr,accuracy)
plt.xlabel("K")
plt.ylabel("Acuracy")
plt.show()
