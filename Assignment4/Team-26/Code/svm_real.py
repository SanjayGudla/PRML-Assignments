import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from plots import plots
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
from LDA import LDA
from pca import PCA

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
X_dev_coast = extractData(directories_dev[0])
X_dev_forest = extractData(directories_dev[1])
X_dev_highway = extractData(directories_dev[2])
X_dev_mountain = extractData(directories_dev[3])
X_dev_opencountry = extractData(directories_dev[4])

x_train = X_train_coast+X_train_forest+X_train_highway+X_train_mountain+X_train_opencountry
c_train = [1]*len(X_train_coast)+[2]*len(X_train_forest)+[3]*len(X_train_highway)+[4]*len(X_train_mountain)+[5]*len(X_train_opencountry)
x_test = X_dev_coast+X_dev_forest+X_dev_highway+X_dev_mountain+X_dev_opencountry
c_test = [1]*len(X_dev_coast)+[2]*len(X_dev_forest)+[3]*len(X_dev_highway)+[4]*len(X_dev_mountain)+[5]*len(X_dev_opencountry)

scl=MinMaxScaler().fit(x_train)
x_train=scl.transform(x_train)
x_test=scl.transform(x_test)
X_train_coast = scl.transform(X_train_coast)
X_train_forest = scl.transform(X_train_forest)
X_train_highway = scl.transform(X_train_highway)
X_train_mountain = scl.transform(X_train_mountain)
X_train_opencountry =scl.transform(X_train_opencountry)

X_trains = []
X_tests = []
X_trains.append(x_train)
X_tests.append(x_test)

################################################################################################
### With PCA
pca_obj = PCA(x_train,828,80)
reduced_train = pca_obj.pca_func()
reduced_dev = pca_obj.project_test_array(x_test)
X_trains.append(reduced_train)
X_tests.append(reduced_dev)
#################################################################################################

#######Applying LDA after PCA
X_train = []
X_train.append(np.array(X_train_coast))
X_train.append(np.array(X_train_forest))
X_train.append(np.array(X_train_highway))
X_train.append(np.array(X_train_mountain))
X_train.append(np.array(X_train_opencountry))
X_train = pca_obj.project_data(X_train)
X_train = np.array(X_train)
X_dev1= (pca_obj.project_test_array(x_test))
lda_obj = LDA(X_train,80,4)
reduced_train= lda_obj.lda_func()
reduced_dev = lda_obj.project_test_array(X_dev1)
X_trains.append(reduced_train)
X_tests.append(reduced_dev)
#################################################################################################

accuracy = []
fpr_arr = []
tpr_arr = []

for (x_train,x_test) in zip(X_trains,X_tests):
    model = SVC(kernel="rbf", C=20,probability = True)
    model.fit(x_train,c_train)
    prediction = model.predict(x_test)
    scores = model.predict_proba(x_test)
    p = list(prediction)
    count = 0
    for i in range(len(p)):
        if p[i] == c_test[i]:
            count += 1
    print(f"Accuracy is : {count / len(p) * 100} %")
    accuracy.append(count / len(c_test) * 100)
    (fpr,tpr) = plots.plot_ROC_curve(np.array(scores),len(c_test),5,c_test)
    plots.draw_confusion(prediction,5,c_test)
    plt.show()
    fpr_arr.append(fpr)
    tpr_arr.append(tpr)


for it in range(len(fpr_arr)):
    label = "SVM"
    if it==1:
        label = label+" With PCA"
    elif it==2:
        label = label+" PCA and LDA"
    plt.plot(fpr_arr[it],tpr_arr[it],label=label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
plt.legend()
plt.show()


fig, ax_det = plt.subplots(1,1)
for it in range(len(fpr_arr)):
    label = "SVM"
    if it==1:
        label = label+" With PCA"
    elif it==2:
        label = label+" PCA and LDA"
    d1=DetCurveDisplay(fpr=fpr_arr[it],fnr=1-tpr_arr[it],estimator_name=label).plot(ax=ax_det)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET  Curve")
plt.show()
