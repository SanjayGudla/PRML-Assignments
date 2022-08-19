import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay


# with open("roc_det1.txt","a") as f:
#     s = ""
#     for x in fpr:
#         s = s+str(x)+" "
#     f.write(s)
#     f.write("\n")
#     s = ""
#     for x in tpr:
#         s = s+str(x)+" "
#     f.write(s)
#     f.write("\n")
#
# plt.plot(fpr,tpr)
# plt.show()

f = open("roc_det.txt","r")
## f = open("roc_det1.txt","r") - RealData

lines = f.readlines()

tpr = []
fpr=[]
fnr = []
for i in range(0,10,2):
    line1 = lines[i].split()
    line1 = list(map(float,line1))
    line2 = lines[i+1].split()
    line2 = list(map(float,line2))
    fpr.append(np.array(line1))
    tpr.append(np.array(line2))

for i in range(5):
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.plot(fpr[i],tpr[i],label="case "+str(i+1))

plt.legend()
plt.show()

fig,ax_det = plt.subplots(1,1)
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
x = DetCurveDisplay(fpr=fpr[0],fnr=1-tpr[0],estimator_name="case 1").plot(ax=ax_det)
x = DetCurveDisplay(fpr=fpr[1],fnr=1-tpr[1],estimator_name="case 2").plot(ax=ax_det)
x = DetCurveDisplay(fpr=fpr[2],fnr=1-tpr[2],estimator_name="case 3").plot(ax=ax_det)
x = DetCurveDisplay(fpr=fpr[3],fnr=1-tpr[3],estimator_name="case 4").plot(ax=ax_det)
x = DetCurveDisplay(fpr=fpr[4],fnr=1-tpr[4],estimator_name="case 5").plot(ax=ax_det)
plt.legend()
plt.show()
