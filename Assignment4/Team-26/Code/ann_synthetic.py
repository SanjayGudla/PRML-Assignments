import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
from plots import plots


f1 = "./Synthetic_Data/26/train.txt"
f2 = "./Synthetic_Data/26/dev.txt"

def fetch_data(f):
    file = open(f, "r")
    X1 = []
    X2 = []
    C = []
    X = []
    while True:
        line = file.readline()
        if not line:
            break
        x, y, c = line.strip().split(',')
        if c == '1':
            X1.append((np.longdouble(x), np.longdouble(y)))
        elif c == '2':
            X2.append((np.longdouble(x), np.longdouble(y)))
        C.append(int(c))
        X.append((np.longdouble(x), np.longdouble(y)))

    X1 = np.array(X1)
    X2 = np.array(X2)
    return (C,X1,X2, X)

c_train,x1_train,x2_train, x_train = fetch_data(f1)
c_test,x1_test,x2_test, x_test = fetch_data(f2)

def Plot_scatter(X1, X2):
    (A, B) = X1[:,0],X1[:,1]
    (C, D) = X2[:,0],X2[:,1]
    plt.scatter(A, B, c="blue")
    plt.scatter(C, D, c="red")
    plt.suptitle("Decision Boundaries with countours")
    plt.title("Blue - Class1 , Red - Class2")
    plt.xlabel("dimension1")
    plt.ylabel("dimension2")

#####################################################################################
accuracy = []
fpr_arr = []
tpr_arr = []
C_s = [5,10,15,20]

for C in C_s:
    clf = MLPClassifier(random_state=C, max_iter=5000).fit(x_train, c_train)
    prediction = clf.predict(x_test)
    scores = clf.predict_proba(x_test)
    p = list(prediction)
    count = 0
    for i in range(len(p)):
        if p[i] == c_test[i]:
            count += 1
    print(f"Accuracy is : {count / len(p) * 100} %")
    accuracy.append(count / len(c_test) * 100)
    (fpr,tpr) = plots.plot_ROC_curve(np.array(scores),len(c_test),2,c_test)
    fpr_arr.append(fpr)
    tpr_arr.append(tpr)


for it in range(len(fpr_arr)):
    plt.plot(fpr_arr[it],tpr_arr[it],label=("k = "+str(C_s[it])))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
plt.legend()
plt.show()


fig, ax_det = plt.subplots(1,1)
for it in range(len(fpr_arr)):
    d1=DetCurveDisplay(fpr=fpr_arr[it],fnr=1-tpr_arr[it],estimator_name=("k = "+str(C_s[it]))).plot(ax=ax_det)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET  Curve")
plt.show()

plt.plot(C_s,accuracy)
plt.xlabel("K")
plt.ylabel("Acuracy")
plt.show()
