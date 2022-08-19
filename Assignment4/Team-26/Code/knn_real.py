from knn_module import KNN
from pca import PCA
from LDA import LDA
import numpy as np
import matplotlib.pyplot as plt
from plots import plots
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
import os
from sklearn.preprocessing import MinMaxScaler

def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        fig = []
        for line in f.readlines():
            x=line.split()
            x = list(map(float,x))
            fig.extend(x)
        data.append(fig)
    return data

directories_train = ["./Features/coast/train","./Features/forest/train","./Features/highway/train","./Features/mountain/train","./Features/opencountry/train"]
directories_dev = ["./Features/coast/dev","./Features/forest/dev","./Features/highway/dev","./Features/mountain/dev","./Features/opencountry/dev"]
X_train_coast = extractData(directories_train[0])
X_train_forest = extractData(directories_train[1])
X_train_highway = extractData(directories_train[2])
X_train_mountain = extractData(directories_train[3])
X_train_opencountry = extractData(directories_train[4])
X_dev_coast = extractData(directories_dev[0])
X_dev_forest = extractData(directories_dev[1])
X_dev_highway = extractData(directories_dev[2])
X_dev_mountain = extractData(directories_dev[3])
X_dev_opencountry = extractData(directories_dev[4])

X_train = X_train_coast+X_train_forest+X_train_highway+X_train_mountain+X_train_opencountry
C_train = [1]*len(X_train_coast)+[2]*len(X_train_forest)+[3]*len(X_train_highway)+[4]*len(X_train_mountain)+[5]*len(X_train_opencountry)
X_dev = X_dev_coast+X_dev_forest+X_dev_highway+X_dev_mountain+X_dev_opencountry
C_dev = [1]*len(X_dev_coast)+[2]*len(X_dev_forest)+[3]*len(X_dev_highway)+[4]*len(X_dev_mountain)+[5]*len(X_dev_opencountry)
#####################################################################################################################

scl=MinMaxScaler().fit(X_train)
x_train=scl.transform(X_train)
X_train_coast = scl.transform(X_train_coast)
X_train_forest = scl.transform(X_train_forest)
X_train_highway = scl.transform(X_train_highway)
X_train_mountain = scl.transform(X_train_mountain)
X_train_opencountry =scl.transform(X_train_opencountry)
x_test=scl.transform(X_dev)
#####################################################################################################################
knn_k = [7,10,15,30,50,70,90,100]
fpr1 = []
tpr1 = []
def plot_roc_det(reduced_train,reduced_dev):
    accuracy = []
    fpr_arr = []
    tpr_arr = []
    for ks in knn_k:
        knn_obj = KNN(reduced_train,reduced_dev,C_train,C_dev,5,ks)
        knn_obj.compute_distance()
        prediction,scores  = knn_obj.classify()
        count = 0
        for i in range(len(C_dev)):
            if C_dev[i] == prediction[i]:
                count += 1
        accuracy.append(count / len(C_dev) * 100)
        print(count / len(C_dev) * 100)
        (fpr,tpr) = plots.plot_ROC_curve(np.array(scores),len(C_dev),5,C_dev)
        if ks==15:
            plots.draw_confusion(prediction,5,C_dev)
            plt.show()
        if ks==70:
            tpr1.append(tpr)
            fpr1.append(fpr)
        fpr_arr.append(fpr)
        tpr_arr.append(tpr)

    for it in range(len(fpr_arr)):
        plt.plot(fpr_arr[it],tpr_arr[it],label=("k = "+str(knn_k[it])))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
    plt.legend()
    plt.show()

    fig, ax_det = plt.subplots(1,1)
    for it in range(len(fpr_arr)):
        d1=DetCurveDisplay(fpr=fpr_arr[it],fnr=1-tpr_arr[it],estimator_name=("k = "+str(knn_k[it]))).plot(ax=ax_det)
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.title("DET  Curve")
    plt.show()



#####################################################################################################################
#### Without LDA and PCA
plot_roc_det(X_train,X_dev)
################################################################################################
### With PCA
pca_obj = PCA(x_train,828,80)
reduced_train = pca_obj.pca_func()
reduced_dev = pca_obj.project_test_array(x_test)
plot_roc_det(reduced_train,reduced_dev)
#################################################################################################
# #LDA  followed by PCA
X_train = []
X_train.append(np.array(X_train_coast))
X_train.append(np.array(X_train_forest))
X_train.append(np.array(X_train_highway))
X_train.append(np.array(X_train_mountain))
X_train.append(np.array(X_train_opencountry))
X_train = np.array(X_train)
X_train= (pca_obj.project_data(X_train))
X_dev1= (pca_obj.project_test_array(x_test))
lda_obj = LDA(X_train,80,4)
reduced_train = lda_obj.lda_func()
reduced_dev = lda_obj.project_test_array(X_dev1)
plot_roc_det(reduced_train,reduced_dev)

################################################################################################
#LDA
# X_train = []
# X_train.append(np.array(X_train_coast))
# X_train.append(np.array(X_train_forest))
# X_train.append(np.array(X_train_highway))
# X_train.append(np.array(X_train_mountain))
# X_train.append(np.array(X_train_opencountry))
# X_train = np.array(X_train)
# lda_obj = LDA(X_train,828,4)
# reduced_train = lda_obj.lda_func()
# reduced_dev = lda_obj.project_test_array(x_test)
# plot_roc_det(reduced_train,reduced_dev)
##########################################################################################################################################

for it in range(len(fpr1)):
    label = "KNN"
    if it==1:
        label=label+" PCA"
    if it==2:
        label =label+" PCA and LDA"
    plt.plot(fpr1[it],tpr1[it],label=label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1,1)
for it in range(len(fpr_arr)):
    label = "KNN"
    if it==1:
        label=label+" PCA"
    if it==2:
        label =label+" PCA and LDA"
    d1=DetCurveDisplay(fpr=fpr1[it],fnr=1-tpr1[it],estimator_name=label).plot(ax=ax_det)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET  Curve")
plt.show()
