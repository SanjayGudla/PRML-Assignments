################################################################################################################################################################
# Importiong Libraries
import numpy as np
import math
# from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random

# Own created classes
from kmeans_n_c import gen_kmeans
from gmm_n_c import gen_gmm

########################################## Bayesian Classification #############################################
##Fetching Data

f1 = "train.txt"
f2 = "dev.txt"

def split_X_Y(X):
    P = np.array([x for (x, y) in X])
    Q = np.array([y for (x, y) in X])
    return (P, Q)

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
    return (C, X1, X2, X)

def Normalize(X):
    n = len(X)
    (P,Q) = split_X_Y(X)
    mean =(np.sum(P)/n,np.sum(Q)/n)
    var1 = np.sum((x-mean[0])**2 for x in P) ** 0.5
    var2 = np.sum((y-mean[1])**2 for y in Q) ** 0.5
    X1 = []
    for (x,y) in X:
        X1.append(( (x-mean[0])/var1, (y-mean[1]/var2)))
    return np.array(X1)

##global variables
(C, X1, X2, X) = fetch_data(f1)
(C_dev, X1_dev, X2_dev, X_dev) = fetch_data(f2)
# X1,X1_dev,X = Normalize(X1),Normalize(X1_dev),Normalize(X)
# X2,X2_dev,X_dev = Normalize(X2),Normalize(X2_dev),Normalize(X_dev)
n1 = np.size(X_dev) // 2
scores_kmeans = np.array([[0.0 for j in range(2)] for i in range(n1)])
scores_gmm = np.array([[0.0 for j in range(2)] for i in range(n1)])
C_dev_found_kmeans=[]
C_dev_found_gmm=[]
#####################################################################################
##Helper functions


def split_X_Y1(X):
    P = np.array([x for (x, y) in X])
    X_min = np.min(P)
    X_max = np.max(P)
    P = np.arange(X_min, X_max, 0.1)
    Q = np.array([y for (x, y) in X])
    Y_min = np.min(Q)
    Y_max = np.max(Q)
    Q = np.arange(Y_min, Y_max, 0.1)
    return P, Q
#####################################################################################
###prediction for kmeans
def predict_class_kmeans(mu1,mu2,k,point,it):
    distance_array1 = [math.dist(point, mu1[i]) for i in range(k)]
    min1 = min(distance_array1)
    distance_array2 = [math.dist(point, mu2[i]) for i in range(k)]
    min2 = min(distance_array2)
    if it!=-1:
        scores_kmeans[it][0] = min2
        scores_kmeans[it][1] = min1
    if(min1<min2):
        return 1
    else:
        return 2

def calc_accuray_kmeans(mu1,mu2,k):
    it = 0
    for point in X_dev:
        C_dev_found_kmeans.append(predict_class_kmeans(mu1,mu2,k,point,it))
        it+=1
    count1 = 0
    for i in range(np.size(C_dev)):
        if (C_dev_found_kmeans[i] == C_dev[i]):
            count1 = count1 + 1
    print("Accuracy for K-Means :- " + str(count1 / np.size(C_dev) * 100))
#################################################################################
##prediction for gmm
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

def predict_class_gmm(mu1,mu2,pi1,pi2,sigma1,sigma2,k,point,it):
    distance_array1 = [calculate_gamma_nk(pi1[i],mu1[i],sigma1[i],point) for i in range(k)]
    min1 = np.sum(np.array(distance_array1))
    distance_array2 = [calculate_gamma_nk(pi2[i],mu2[i],sigma2[i],point) for i in range(k)]
    min2 = np.sum(np.array(distance_array2))
    if it!=-1:
        scores_gmm[it][0] = min1
        scores_gmm[it][1] = min2
    if(min1>min2):
        return 1
    else:
        return 2

def calc_accuracy_gmm(mu1,mu2,pi1,pi2,sigma1,sigma2,k):
    it=0
    for point in X_dev:
        C_dev_found_gmm.append(predict_class_gmm(mu1,mu2,pi1,pi2,sigma1,sigma2,k,point,it))
        it+=1
    count1 = 0
    for i in range(np.size(C_dev)):
        if (C_dev_found_gmm[i] == C_dev[i]):
            count1 = count1 + 1
    print("Accuracy for GMM :- " + str(count1 / np.size(C_dev) * 100))
#################################################################################
### Plots
def Plot_scatter(X1, X2):
    (A, B) = X1[:,0],X1[:,1]
    (C, D) = X2[:,0],X2[:,1]
    plt.scatter(A, B, c="blue")
    plt.scatter(C, D, c="red")
    plt.suptitle("Decision Boundaries with countours")
    plt.title("Blue - Class1 , Red - Class2")
    plt.xlabel("dimension1")
    plt.ylabel("dimension2")

def Plot_PDF(X,mu1, sigma1):
    distr = multivariate_normal(cov=sigma1, mean=mu1)
    (p,q) = split_X_Y(X)
    x = np.arange(np.min(p),np.max(p),0.1)
    y = np.arange(np.min(q),np.max(q),0.1)
    X, Y = np.meshgrid(x, y)
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
    return (X, Y, pdf)


def Plot_Contour(X, mu1, sigma1):
    (A, B, pdf) = Plot_PDF(X,mu1, sigma1)
    pdf_list = []
    pdf_list.append(pdf)
    for idx, val in enumerate(pdf_list):
        plt.contour(A, B, val)
        plt.xlabel("Dimension - 1")
        plt.ylabel("Dimension - 2")

def Plot_Decision_kmeans(X,mu1,mu2,k):
    (p,q) = split_X_Y(X)
    x1grid = np.arange(np.min(p)-1,np.max(p)+1,0.05)
    x2grid = np.arange(np.min(q)-1,np.max(q)+1,0.05)
    xx, yy = np.meshgrid(x1grid, x2grid)
    yhat =[]
    for y in x2grid:
        for x in x1grid:
            yhat.append(predict_class_kmeans(mu1,mu2,k,(x,y),-1))
    yhat = np.array(yhat)
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')

def Plot_Decision_gmm(X,mu1,mu2,pi1,pi2,sigma1,sigma2,k):
    (p,q) = split_X_Y(X)
    x1grid = np.linspace(np.min(p)-1,np.max(p)+1,100)
    x2grid = np.linspace(np.min(q)-1,np.max(q)+1,100)
    xx, yy = np.meshgrid(x1grid, x2grid)
    yhat =[]
    for y in x2grid:
        for x in x1grid:
            yhat.append(predict_class_gmm(mu1,mu2,pi1,pi2,sigma1,sigma2,k,(x,y),-1))
    yhat = np.array(yhat)
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')


def plot_ROC_curve(scores, n,s):
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
            for j in range(2):
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


def draw_confusion(res1):
    conf= np.array([[0 for j in range(2)] for i in range(2)])
    for i in range(np.size(res1)):
        conf[C_dev[i]-1][res1[i]-1] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')

############################################################################################
k = int(input("Please Enter Number of Clusters :-"))
rand_val1 = random.sample(range(len(X1)),k)
rand_val2 = random.sample(range(len(X2)),k)
def Kmeans_Gmm(D):
    string = "Diag"
    if D==0 :
        string = "Non-Diag"
    km_obj1 = gen_kmeans(X1,2,k,2,D,rand_val1)
    km_out1 = km_obj1.kmeans_fun()
    mu1,sigma1,pi1,cluster1 = km_out1
    km_obj2 = gen_kmeans(X2,2,k,2,D,rand_val2)
    km_out2 = km_obj2.kmeans_fun()
    mu2,sigma2,pi2,cluster2 = km_out2
    calc_accuray_kmeans(mu1,mu2,k)
    (mu_x, mu_y) = split_X_Y(mu1)
    (mu_x2, mu_y2) = split_X_Y(mu2)
    plt.subplot(2,2,2*D+1)
    for i in range(k):
        Plot_Contour(cluster1[i],mu1[i],sigma1[i])
        Plot_Contour(cluster2[i],mu2[i],sigma2[i])
    Plot_Decision_kmeans(X,mu1,mu2,k)
    Plot_scatter(X1_dev, X2_dev)
    plt.scatter(mu_x, mu_y, c="orange")
    plt.scatter(mu_x2, mu_y2, c="violet")
    plt.title("Kmeans for k = "+str(k)+" for "+string)

    gmm_obj1 = gen_gmm(X1,2,k,2,km_out1,D)
    gmm_obj2 = gen_gmm(X2,2,k,2,km_out2,D)
    mu1,sigma1,pi1 = gmm_obj1.gmm_fun()
    mu2,sigma2,pi2 = gmm_obj2.gmm_fun()
    calc_accuracy_gmm(mu1,mu2,pi1,pi2,sigma1,sigma2,k)
    (mu_x, mu_y) = split_X_Y(mu1)
    (mu_x2, mu_y2) = split_X_Y(mu2)
    plt.subplot(2,2,2*D+2)
    for i in range(k):
        Plot_Contour(cluster1[i],mu1[i],sigma1[i])
        Plot_Contour(cluster2[i],mu2[i],sigma2[i])
    Plot_Decision_gmm(X,mu1,mu2,pi1,pi2,sigma1,sigma2,k)
    Plot_scatter(X1_dev, X2_dev)
    plt.scatter(mu_x, mu_y, c="orange")
    plt.scatter(mu_x2, mu_y2, c="violet")
    plt.title("GMM for k = "+str(k)+" for "+string)

plt.figure(figsize = (14,8))
Kmeans_Gmm(0)
scores_kmeans_nondiag = scores_kmeans
scores_gmm_nondiag = scores_gmm
C_dev_found_kmeans_nondiag = C_dev_found_kmeans
C_dev_found_gmm_nondiag = C_dev_found_gmm


scores_kmeans = np.array([[0.0 for j in range(2)] for i in range(n1)])
scores_gmm = np.array([[0.0 for j in range(2)] for i in range(n1)])
C_dev_found_kmeans=[]
C_dev_found_gmm=[]
Kmeans_Gmm(1)
scores_kmeans_diag = scores_kmeans
scores_gmm_diag = scores_gmm
C_dev_found_kmeans_diag = C_dev_found_kmeans
C_dev_found_gmm_diag = C_dev_found_gmm
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()

plt.figure(figsize = (14,8))
plt.subplot(2,2,1)
draw_confusion(C_dev_found_kmeans_nondiag)
plt.title("Confusion Matrix for Kmeans with k = "+str(k),y=-0.3)
plt.subplot(2,2,2)
draw_confusion(C_dev_found_gmm_nondiag)
plt.title("Confusion Matrix for GMM with k = "+str(k),y=-0.3)
plt.subplot(2,2,3)
draw_confusion(C_dev_found_kmeans_diag)
plt.title("Confusion Matrix for GMM with k = "+str(k),y=-0.3)
plt.subplot(2,2,4)
draw_confusion(C_dev_found_gmm_diag)
plt.title("Confusion Matrix for GMM with k = "+str(k),y=-0.3)
plt.show()


(fpr1,tpr1)=plot_ROC_curve(scores_kmeans_nondiag, n1,"kmeans_nondiag")
(fpr2,tpr2)=plot_ROC_curve(scores_kmeans_diag, n1,"kmeans_diag")
(fpr3,tpr3)=plot_ROC_curve(scores_gmm_nondiag, n1,"gmm_nondiag")
(fpr4,tpr4)=plot_ROC_curve(scores_gmm_diag, n1,"gmm_diag")
plt.legend()
plt.show()

fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=1-tpr1,estimator_name="case 1-kmeans_nondiag").plot(ax=ax_det)
d2=DetCurveDisplay(fpr=fpr2,fnr=1-tpr2,estimator_name="case 2-kmeans_diag").plot(ax=ax_det)
d3=DetCurveDisplay(fpr=fpr3,fnr=1-tpr3,estimator_name="case 3-gmm_nondiag").plot(ax=ax_det)
d4=DetCurveDisplay(fpr=fpr4,fnr=1-tpr4,estimator_name="case 4-gmm_diag").plot(ax=ax_det)
plt.show()
