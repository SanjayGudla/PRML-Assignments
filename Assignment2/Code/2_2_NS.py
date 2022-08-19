################################################################################################################################################################
# Importiong Libraries
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
fig,ax_det = plt.subplots(1,1,figsize=(20,10))

from scipy.stats import multivariate_normal

########################################## Bayesian Classification #############################################
##Fetching Data

f1 = "trian.txt"
f2 = "dev.txt"
def fetch_data(f):
    file = open(f,"r")
    X1 = []
    X2 = []
    X3 = []
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
        elif c == '3':
            X3.append((np.longdouble(x), np.longdouble(y)))
        C.append(int(c))
        X.append(np.longdouble((x, y)))
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    return (C, X1, X2, X3, X)

def split_X_Y(X):
    P = np.array([x for (x,y) in X])
    Q = np.array([y for (x,y) in X])
    return (P, Q)

def split_X_Y1(X):
    P = np.array([x for (x,y) in X])
    X_min = np.min(P)
    X_max = np.max(P)
    P = np.arange(X_min,X_max,0.1)
    Q = np.array([y for (x,y) in X])
    Y_min = np.min(Q)
    Y_max = np.max(Q)
    Q = np.arange(Y_min,Y_max,0.1)
    return (P, Q)

def calculate_mu_sigma(c):
    sizeofclass = np.size(c) // 2
    (X,Y)= split_X_Y(c)
    x_mean = np.sum(X)
    y_mean = np.sum(Y)
    (x_mean, y_mean) = (x_mean / sizeofclass, y_mean / sizeofclass)
    sigma = np.array([[0., 0.], [0., 0.]])
    for i in range(0, sizeofclass):
        (a, b) = c[i] - (x_mean, y_mean)
        sigma[0][0] += a * a
        sigma[0][1] += a * b
        sigma[1][0] += a * b
        sigma[1][1] += b * b
    sigma = sigma / (sizeofclass - 1)
    return ((x_mean, y_mean), sigma)

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
            for j in range(3):
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
    for i in range(n):
        x = max(scores[i][0], scores[i][1])
        x = max(x, scores[i][2])
        fnr = np.append(fnr, x)
    plt.plot(fpr, tpr)
    plt.show()


(C_train, X1_train, X2_train, X3_train, X_train) = fetch_data(f1)
(C_dev, X1_dev, X2_dev, X3_dev, X_dev) = fetch_data(f2)

(mu1_train, sigma1_train) = calculate_mu_sigma(X1_train)
(mu2_train, sigma2_train) = calculate_mu_sigma(X2_train)
(mu3_train, sigma3_train) = calculate_mu_sigma(X3_train)
(nu,sigma_total)=calculate_mu_sigma(X_train)
# ########################################################################################################################################
# ##case 2:- Bayes with C1 != C2 != C3.

def calculate_prob(c, x, y):
    ((p, q), sigma) = calculate_mu_sigma(c)
    (a, b) = (x - p, y - q)
    t = np.row_stack([[a, b]])
    g = 0
    g -= np.log(np.sqrt(2 * math.pi)) / 2
    g -= np.log(math.sqrt(LA.det(sigma)))
    g -= t @ LA.inv(sigma) @ (t.T)
    g = g / 2
    return g

n = np.size(X_dev) // 2
res = []
it = 0
scores = np.array([[0.0 for j in range(3)] for i in range(n)])
for (x, y) in X_dev:
    g1 = calculate_prob(X1_train, x, y)
    g2 = calculate_prob(X2_train, x, y)
    g3 = calculate_prob(X3_train, x, y)
    if (g1 >= g2 and g1 >= g3):
        res.append(1)
    elif g2 >= g3:
        res.append(2)
    else:
        res.append(3)
    scores[it][0] = math.exp(g1)/3
    scores[it][1] = math.exp(g2)/3
    scores[it][2] = math.exp(g3)/3
    it = it + 1

count = 0
for i in range(np.size(res)):
    if (res[i] == C_dev[i]):
        count = count + 1
print("case 2-Bayes with Covariance different for all classes:- " + str(count / np.size(res) * 100))
###########################################################################################################
              ###########  plots ###########

(A, B) = split_X_Y(X1_train)
(C, D) = split_X_Y(X2_train)
(E, F) = split_X_Y(X3_train)


def plot_decision(mu1,mu2,Sigma1,Sigma2,m,n):
    (p,q)=mu1
    (r,s)=mu2
    MU1t = np.array([p,q])
    MU2t = np.array([r,s])
    MU1=MU1t.T
    MU2=MU2t.T
    W = LA.inv(Sigma1)-LA.inv(Sigma2)
    a = -1*W[0][0]/2
    b = -1*W[0][1]/2
    c = -1*W[1][0]/2
    d = -1*W[1][1]/2
    w = LA.inv(Sigma1)@MU1 - LA.inv(Sigma2)@MU2
    e = w[0]
    f = w[1]
    w0 = MU2t@LA.inv(Sigma2)@MU2 + MU1t@LA.inv(Sigma1)@MU1
    w0 = w0 + np.log(LA.det(Sigma1))-np.log(LA.det(Sigma2))
    g =-1*w0/2
    x = np.arange(m,n,0.1)
    y = np.arange(m,n,0.1)
    X,Y = np.meshgrid(x,y)
    F = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            F[i,j] = a*X[i,j]*X[i,j]+d*Y[i,j]*Y[i,j]+(b+c)*X[i,j]*Y[i,j]+e*X[i,j]+f*Y[i,j]+g
    return (X,Y,F)

def plot_decisions(mu1,sigma1,mu2,sigma2,mu3,sigma3):
    (A,B,pdf)=plot_decision(mu1,mu2,sigma1,sigma2,-40,40)
    pdf_list = []
    pdf_list.append(pdf)
    for idx, val in enumerate(pdf_list):
    	plt.contour(A, B, val, levels=[0])
    	plt.xlabel("x1")
    	plt.ylabel("x2")
    (C,D,pdf1)=plot_decision(mu2,mu3,sigma2,sigma3,-40,40)
    pdf_list1 = []
    pdf_list1.append(pdf1)
    for idx, val in enumerate(pdf_list1):
    	plt.contour(C, D, val, levels=[0])
    	plt.xlabel("x1")
    	plt.ylabel("x2")
    (E,F,pdf2)=plot_decision(mu1,mu3,sigma1,sigma3,-40,40)
    pdf_list2 = []
    pdf_list2.append(pdf2)
    for idx, val in enumerate(pdf_list2):
    	plt.contour(E, F, val, levels=[0])
    	plt.xlabel("x1")
    	plt.ylabel("x2")

def Plot_PDF(X1,mu1,sigma1):
    distr = multivariate_normal(cov = sigma1, mean = mu1)
    (x,y) = split_X_Y1(X1)
    X, Y = np.meshgrid(x,y)
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
    return (X,Y,pdf)

def Plot_Contour(X1,mu1,sigma1):
    (A,B,pdf)=Plot_PDF(X1,mu1,sigma1)
    pdf_list = []
    pdf_list.append(pdf)
    for idx, val in enumerate(pdf_list):
    	plt.contour(A, B, val, cmap='viridis')
    	plt.xlabel("x1")
    	plt.ylabel("x2")

def Plot_PDFs(X1,mu1,sigma1,X2,mu2,sigma2,X3,mu3,sigma3):
    ax = plt.axes(projection='3d')
    (A,B,pdf)=Plot_PDF(X1,mu1,sigma1)
    ax.scatter3D(A,B,pdf,cmap='viridis')
    (A,B,pdf)=Plot_PDF(X2,mu2,sigma2)
    ax.scatter3D(A,B,pdf,cmap='viridis')
    (A,B,pdf)=Plot_PDF(X3,mu3,sigma3)
    ax.scatter3D(A,B,pdf,cmap='viridis')


def Plot_Contours(X1,mu1,sigma1,X2,mu2,sigma2,X3,mu3,sigma3):
    Plot_Contour(X1,mu1,sigma1)
    Plot_Contour(X2,mu2,sigma2)
    Plot_Contour(X3,mu3,sigma3)


def Plot_scatter(X1,X2,X3):
    (A, B) = split_X_Y(X1)
    (C, D) = split_X_Y(X2)
    (E, F) = split_X_Y(X3)
    plt.scatter(A, B, c="blue")
    plt.scatter(C, D, c="red")
    plt.scatter(E, F, c="green")
    plt.suptitle("Given data")
    plt.title("Blue - class1, Red - class2, Green - class3")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

def Plot_EigenVetors(mu1,sigma1):
    (p,q)=mu1
    MU = [p,q]
    w,v = LA.eig(sigma1)
    e1 = v[:,0]
    e2 = v[:,1]
    plt.quiver(*MU,*e1,scale=10)
    plt.quiver(*MU,*e2,scale=10)

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
            for j in range(3):
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
        x = max(scores[i][0], scores[i][1])
        x = max(x, scores[i][2])
        fnr = np.append(fnr, x)
    # plt.plot(fpr, tpr)
    # plt.show()
    return(fpr,tpr,fnr)

################################################################################################################
#1 - pdf of given data
Plot_PDFs(X1_train,mu1_train,sigma1_train,X2_train,mu2_train,sigma2_train,X3_train,mu3_train,sigma3_train)
plt.show()
Plot_Contours(X1_train,mu1_train,sigma1_train,X2_train,mu2_train,sigma2_train,X3_train,mu3_train,sigma3_train)
plt.show()


# #2-scatter+decision+contour
Plot_Contours(X1_train,mu1_train,sigma1_train,X2_train,mu2_train,sigma2_train,X3_train,mu3_train,sigma3_train)
#plot_decisions(mu1_train,sigma1_train,mu2_train,sigma2_train,mu3_train,sigma3_train)
Plot_scatter(X1_train,X2_train,X3_train)
plt.show()

#3  ROC and Det
(fpr,tpr,fnr)=plot_ROC_curve(scores, n)

with open("roc_det.txt","a") as f:
    s = ""
    for x in fpr:
        s = s+str(x)+" "
    f.write(s)
    f.write("\n")
    s = ""
    for x in tpr:
        s = s+str(x)+" "
    f.write(s)
    f.write("\n")

plt.plot(fpr,tpr)
plt.show()
display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: 2").plot(ax=ax_det)


#eigenvectors
Plot_Contours(X1_train,mu1_train,sigma1_train,X2_train,mu2_train,sigma2_train,X3_train,mu3_train,sigma3_train)
Plot_EigenVetors(mu1_train,sigma1_train)
Plot_EigenVetors(mu2_train,sigma2_train)
Plot_EigenVetors(mu3_train,sigma3_train)
plt.show()

#confusion Matrix
#Confusion Matrix
conf= np.array([[0 for j in range(3)] for i in range(3)])
for i in range(np.size(res)):
    conf[C_dev[i]-1][res[i]-1] += 1
df_cm = pd.DataFrame(conf)
as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
as1.set_xlabel('true class')
as1.set_ylabel('predicted class')
plt.show()
