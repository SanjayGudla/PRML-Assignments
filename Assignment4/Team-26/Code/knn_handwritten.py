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
##########################################################################################
## Data Extraction and preprocessing
scores = []
lengths = []
def extractData(directory):
    data=[]
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename)
        f=open(f,"r")
        m=[]
        line = f.readline()
        line = line.split()
        num_cordinates = int(line[0])
        lengths.append(num_cordinates)
        num_cordinates = 2*num_cordinates
        x=[]
        y=[]
        for i in range(1,num_cordinates,2):
            l = [line[i],line[i+1]]
            x.append(float(line[i]))
            y.append(float(line[i+1]))
            l=list(map(float,l))
            m.append(l)
        (x_min,x_max) = (np.min(np.array(x)),np.max(np.array(x)))
        (y_min,y_max) = (np.min(np.array(y)),np.max(np.array(y)))
        m_1 = []
        for [p,q] in m:
            p = (p-x_min)/(x_max-x_min)
            q = (q-y_min)/(y_max-y_min)
            m_1.append([p,q])
        data.append(m_1)
    return np.array(data)


directories_train = ["./Handwritten/ai/train","./Handwritten/bA/train","./Handwritten/chA/train","./Handwritten/dA/train","./Handwritten/tA/train"]
directories_dev = ["./Handwritten/ai/dev","./Handwritten/bA/dev","./Handwritten/chA/dev","./Handwritten/dA/dev","./Handwritten/tA/dev"]

X_train_ai = extractData(directories_train[0])
X_train_bA = extractData(directories_train[1])
X_train_chA = extractData(directories_train[2])
X_train_dA = extractData(directories_train[3])
X_train_tA = extractData(directories_train[4])
X_dev_ai = extractData(directories_dev[0])
X_dev_bA = extractData(directories_dev[1])
X_dev_chA = extractData(directories_dev[2])
X_dev_dA = extractData(directories_dev[3])
X_dev_tA= extractData(directories_dev[4])
datapoint_min_length= min(lengths)


def find_avg(data_point,i,j):
    x = np.zeros(2)
    for k in range(i,j):
        x = x+ np.array(data_point[k])
    x = x/(j-i)
    return x

def flatten_features(data_set,index):
    X_flatten = []
    temp = 0;
    for data_point in data_set:
        data_point_new = []
        data_point_length = len(data_point)
        n = data_point_length-datapoint_min_length+1
        for i in range(datapoint_min_length):
            x = find_avg(data_point,i,i+n)
            data_point_new.extend(x)
        temp+=1
        X_flatten.append(np.array(data_point_new))
    return X_flatten


X_train_ai=flatten_features(X_train_ai,0)
X_train_bA=flatten_features(X_train_bA,1)
X_train_chA=flatten_features(X_train_chA,2)
X_train_dA=flatten_features(X_train_dA,3)
X_train_tA=flatten_features(X_train_tA,4)
X_dev_ai=flatten_features(X_dev_ai,5)
X_dev_bA=flatten_features(X_dev_bA,6)
X_dev_chA=flatten_features(X_dev_chA,7)
X_dev_dA=flatten_features(X_dev_dA,8)
X_dev_tA=flatten_features(X_dev_tA,9)

X_train = X_train_ai+X_train_bA+X_train_chA+X_train_dA+X_train_tA
C_train = [1]*len(X_train_ai)+[2]*len(X_train_bA)+[3]*len(X_train_chA)+[4]*len(X_train_dA)+[5]*len(X_train_tA)
X_dev = X_dev_ai+X_dev_bA+X_dev_chA+X_dev_dA+X_dev_tA
C_dev = [1]*len(X_dev_ai)+[2]*len(X_dev_bA)+[3]*len(X_dev_chA)+[4]*len(X_dev_dA)+[5]*len(X_dev_tA)
#####################################################################################################################

scl=MinMaxScaler().fit(X_train)
x_train=scl.transform(X_train)
X_train_ai = scl.transform(X_train_ai)
X_train_bA = scl.transform(X_train_bA)
X_train_chA = scl.transform(X_train_chA)
X_train_dA = scl.transform(X_train_dA)
X_train_tA =scl.transform(X_train_tA)
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
        if ks==30:
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
#plot_roc_det(x_train,x_test)
################################################################################################
### With PCA
pca_obj = PCA(x_train,38,10)
reduced_train = pca_obj.pca_func()
reduced_dev = pca_obj.project_test_array(x_test)
plot_roc_det(reduced_train,reduced_dev)
#################################################################################################
# #LDA  followed by PCA
X_train = []
X_train.append(np.array(X_train_ai))
X_train.append(np.array(X_train_bA))
X_train.append(np.array(X_train_chA))
X_train.append(np.array(X_train_dA))
X_train.append(np.array(X_train_tA))
X_train = np.array(X_train)
X_train= (pca_obj.project_data(X_train))
X_dev1= (pca_obj.project_test_array(x_test))
lda_obj = LDA(X_train,10,4)
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
for it in range(len(fpr1)):
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
