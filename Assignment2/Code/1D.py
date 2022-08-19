################################################################################################################################################################
#Importiong Libraries
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

f1 = "1d_team_26_train.txt"
f2 = "1d_team_26_dev.txt"
########################################## LINEAR  REGRESSION #############################################

#### 1D data

############################## importing data from file#########################################
## X is array of feature variable and T is array of target variable
## read line by line from given train file using readline() and split the line
## append feature variable x to X and target variable t to T
################################################################################################
def fetch_data(f):
    X = []
    T = []
    file = open(f,"r")
    while True:
        line = file.readline()
        if not line:
            break
        x,t = line.strip().split(' ')
        X.append(x)
        T.append(t)
    X = list(map(float,X))
    T = list(map(float,T))
    X = np.array(X)
    T = np.array(T)
    return (X,T)

##############################################################################################

##fiiting train data and getting optimal weight vector
## n is the size of data set , m is the model complexity
def fit_data_train(n,m,l):
    (X,T)=fetch_data(f1)
    target_output = np.array([T[i] for i in range(n)])
    input = np.array([X[i] for i in range(n)])
    phi = np.array([[X[i]**j for j in range(m)]for i in range(n)])
    phi_pseudoinv = (phi.T)@phi
    phi_pseudoinv = phi_pseudoinv + (np.identity(m,dtype=float))*l
    phi_pseudoinv = LA.inv(phi_pseudoinv)
    phi_pseudoinv = phi_pseudoinv @ (phi.T)
    w = phi_pseudoinv @ target_output
    model_output = phi @ w.T
    return (w,input,target_output,model_output)

#using weight vector produced by Training data to estimate output of development data
def fit_data_devlopment(n,m,l):
    (X,T)=fetch_data(f2)
    (w,input,target_output,model_output) = fit_data_train(n,m,l)
    phi = []
    target_output1 = []
    input1 = []
    for i in range(n):
        target_output1.append(T[i])
        input1.append(X[i])
        for j in range(m):
            phi.append(X[i]**j)
    phi = np.array(phi)
    phi = phi.reshape(n,m)
    model_output1 = phi @ w.T
    return (w,input1,target_output1,model_output1)
####################################################################################################
####plots

#plotting given output vs model output
def plot_targetvsmodel(n,m,f):
    if(f==f1):
        (w,input,target_output,model_output) = fit_data_train(n,m,0)
    else:
        (w,input,target_output,model_output) = fit_data_devlopment(n,m,0)
    plt.scatter(target_output,model_output,color='b',label = 'produced')
    plt.suptitle("Data Size: "+str(n)+","+" Model Complexity: "+str(m))
    plt.title(" for "+f+" in Least_Square_Regression")
    plt.xlabel("Target_Output")
    plt.ylabel("Model_Output")
    plt.show()

#ploting approximate function
def plot_approxfun(n,m,f,l):
    if(f==f1):
        (w,input,target_output,model_output) = fit_data_train(n,m,l)
    else:
        (w,input,target_output,model_output) = fit_data_devlopment(n,m,l)
    plt.scatter(input,target_output,color='g',label = 'Target_Output')
    plt.plot(input,model_output,color='r',label = 'Model_Output')

    if(l==0):
        plt.suptitle("Data Size: "+str(n)+","+" Model Complexity: "+str(m))
        plt.title("Approx functions for "+f+" in Least_Square_Regression")
    else:
        plt.suptitle("Data Size: "+str(n)+","+" Model Complexity: "+str(m)+"ln(lambda): "+str(np.log(l)))
        plt.title("Approx functions for "+f+" in Ridge_Regeression")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()

##calculating rmsvalues for model complexities 1-13
def calculate_rmsarray_leastsquare (n,f):
    RMS_Array = []
    M = []
    for m in range(1,13):
        M.append(m)
        if(f==f1):
            (w,input,target_output,model_output) = fit_data_train(n,m,0)
        else:
            (w,input,target_output,model_output) = fit_data_devlopment(n,m,0)
        error_array = (model_output - target_output)**2
        Least_Square_Error = 0
        for i in range(n):
            Least_Square_Error=Least_Square_Error+error_array[i]
        Least_Square_Error=Least_Square_Error/2
        RMS_Error = math.sqrt(2*Least_Square_Error/n)
        RMS_Array.append(RMS_Error)
    RMS_Array = np.array(RMS_Array)
    #RMS_Array = RMS_Array/LA.norm(RMS_Array)
    return RMS_Array
#plotting rmsvalues for model complexities 1-13
def plot_rmserror_leastsquare(n):
    M = []
    for m in range(1,13):
        M.append(m)
    RMS_Array1 = calculate_rmsarray_leastsquare(n,f1)
    RMS_Array2 = calculate_rmsarray_leastsquare(n,f2)
    plt.plot(M[0:11],RMS_Array1[0:11],color='b',label = 'Training')
    plt.plot(M,RMS_Array2,color='r',label = 'Development')
    plt.suptitle("Data Size: "+str(n))
    plt.title("ERMS vs M in Least_Square_Regression")
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.show()

#calculating rmsvalues for ln(lambda) values -15 to 0 for ridge regression
def calculate_rmsarray_ridge (n,m,f):
    RMS_Array = []
    L = []
    for l in range(-15,0):
        L.append(l)
        if(f==f1):
            (w,input,target_output,model_output) = fit_data_train(n,m,np.exp(l))
        elif(f==f2):
            (w,input,target_output,model_output) = fit_data_devlopment(n,m,np.exp(l))
        error_array = (model_output - target_output)**2
        Least_Square_Error = 0
        for i in range(n):
            Least_Square_Error=Least_Square_Error+error_array[i]
        Least_Square_Error=Least_Square_Error/2
        RMS_Error = math.sqrt((2*Least_Square_Error)/n)
        RMS_Array.append(RMS_Error)
    RMS_Array = np.array(RMS_Array)
    # RMS_Array = RMS_Array/LA.norm(RMS_Array)
    return RMS_Array
#plotting rmsvalues for ln(lambda) values -15 to 0 for ridge regression
def plot_rmserror_ridge(n,m):
    L = []
    for l in range(-15,0):
        L.append(l)
    RMS_Array1 = calculate_rmsarray_ridge(n,m,f1)
    RMS_Array2 = calculate_rmsarray_ridge(n,m,f2)
    plt.plot(L,RMS_Array1,color='b',label = 'Training')
    plt.plot(L,RMS_Array2,color='r',label = 'Development')
    plt.suptitle("Data Size: "+str(n))
    plt.title("ERMS vs ln(Lambda) in RidgeRegression")
    plt.xlabel("ln(Lambda)")
    plt.ylabel("ERMS")
    plt.show()

#plots used in report
plot_approxfun(200,1,f1,0)
plot_approxfun(200,3,f1,0)
plot_approxfun(200,7,f1,0)
plot_approxfun(200,11,f1,0)
plot_approxfun(10,10,f1,0)
plot_approxfun(100,10,f1,0)
plot_rmserror_leastsquare(200)
plot_targetvsmodel(200,11,f1)
plot_targetvsmodel(200,11,f2)
plot_approxfun(200,13,f1,0)
plot_approxfun(200,13,f1,np.exp(-4))
plot_rmserror_ridge(200,11)

####################################################################################################################
