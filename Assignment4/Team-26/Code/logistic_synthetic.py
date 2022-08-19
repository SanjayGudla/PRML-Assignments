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
from logistic import Logistic

f1 = "train.txt"
f2 = "dev.txt"

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
            X1.append(np.array((np.longdouble(x),np.longdouble(y))))
        elif c == '2':
            X2.append(np.array(( np.longdouble(x), np.longdouble(y))))
        C.append(int(c))
        X.append(np.array(( np.longdouble(x), np.longdouble(y))))

    X1 = np.array(X1)
    X2 = np.array(X2)
    return (C,X1,X2, X)

c_train,x1_train,x2_train, x_train = fetch_data(f1)
c_test,x1_test,x2_test, x_test = fetch_data(f2)


classified,scores = Logistic(x_train,2,2,x_test,c_train).classify()

count = 0

for i in range(len(classified)):
    #print(classified[i], c_test[i])
    if classified[i] == c_test[i]:
        count += 1

print(f"Accuracy is : {count / len(classified) * 100} %")

(fpr,tpr) = plots.plot_ROC_curve(np.array(scores),len(c_test),2,c_test)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

plots.draw_confusion(classified,2,c_test)
plt.show()

d1=DetCurveDisplay(fpr=fpr,fnr=1-tpr).plot(ax=ax_det)
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("DET  Curve")
plt.show()
