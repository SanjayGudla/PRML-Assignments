import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.stats import multivariate_normal
from sklearn.metrics import DetCurveDisplay
import random
import os
from sklearn.cluster import KMeans

####data extraction
m1 = []
 
num_states = 3
num_symbols = 10
def extractData(directory):
    data=[]
    lens = []
    for filename in os.listdir(directory):
        (filename,extension) = os.path.splitext(filename)
        if extension == '.mfcc':
            f=open(directory+'/'+filename+extension,"r")
            m=[]
            lines = f.readlines()
            for line in lines[1:]:
                x=line.split()
                x = list(map(float, x))
                m.append(x)
                m1.append(x)
            data.append(m)
    return np.array(data)

directories_train = ["./3/train","./5/train","./6/train","./8/train","./z/train"]
directories_dev = ["./3/dev","./5/dev","./6/dev","./8/dev","./z/dev"]
X_train3= extractData(directories_train[0])
X_train5= extractData(directories_train[1])
X_train6= extractData(directories_train[2])
X_train8= extractData(directories_train[3])
X_trainz= extractData(directories_train[4])

X_dev3= extractData(directories_dev[0])
X_dev5= extractData(directories_dev[1])
X_dev6= extractData(directories_dev[2])
X_dev8= extractData(directories_dev[3])
X_devz= extractData(directories_dev[4])

X_dev = np.concatenate((X_dev3,X_dev5,X_dev6,X_dev8,X_devz),axis=0)
kmeans1 = KMeans(n_clusters=10, random_state=0).fit(m1)

def forwardalgo(seq_devpoint,HMM_output,numstates):
    rec,next,rec_symbol,next_symbol = HMM_output
    T = len(seq_devpoint)
    alpha = np.zeros(T+1)
    alpha[0]=1
    for t in range(T):
        palpha = np.zeros(T+1)
        for i in range(numstates):
            palpha+=alpha[i]*rec[i]*rec_symbol[i][seq_devpoint[t]]
            if i>0:
                palpha += alpha[i-1]*next[i-1]*next_symbol[i-1][seq_devpoint[t]]
                
        alpha,palpha = palpha,alpha
    return np.sum(alpha)
    	

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
            for j in range(5):
                if (scores[i][j] >= threshold):
                    if ground_truth == j :
                        tp += 1
                    else:
                        fp += 1
                else:
                    if ground_truth == j :
                        fn += 1
                    else:
                        tn += 1
        tpr = np.append(tpr, tp / (tp + fn))
        fpr = np.append(fpr, fp / (fp + tn))
    plt.plot(fpr,tpr,label = s)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")
    return (fpr,tpr)


def draw_confusion(res1):
    conf= np.array([[0 for j in range(5)] for i in range(5)])
    for i in range(np.size(res1)):
        conf[C_dev[i]][res1[i]] += 1
    df_cm = pd.DataFrame(conf)
    as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
    as1.set_xlabel('true class')
    as1.set_ylabel('predicted class')


def HMM(X_train,numsymbols,kmeans,numstates):
    symbolseq = []
    f = open("temp.hmm.seq", "w")
    for sequence in X_train:
        symbolseqi = kmeans.predict(sequence)
        symbolseq.append(symbolseqi)
        p=""
        for symbol in symbolseqi:
            p=p+str(symbol)+" "
        p=p+"\n"
        f.write(p)
    f.close()
    cmd  = "./train_hmm temp.hmm.seq 100000 "+str(numstates)+" "+str(numsymbols)+" .001"+">dump.txt"
    os.system(cmd)
    f1 = open("temp.hmm.seq.hmm","r")
    lines = f1.readlines()
    f1.close()
    i=2
    rec = []
    next = []
    rec_symbol = []
    next_symbol = []
    for i in range(1,numstates+1):
        line1 = list(map(float,lines[(3*i)-1].split()))
        line2 = list(map(float,lines[(3*i)].split()))
        rec.append(line1[0])
        next.append(line2[0])
        rec_symbol.append(line1[1:])
        next_symbol.append(line2[1:])
    return (rec,next,rec_symbol,next_symbol)

C_dev = [0]*len(X_dev3)+[1]*len(X_dev5)+[2]*len(X_dev6)+[3]*len(X_dev8)+[4]*len(X_devz)

def tempfun(numsymbols,numstates):
    kmeans = KMeans(n_clusters=numsymbols, random_state=0).fit(m1)
    HMM_output3 = HMM(X_train3,numsymbols,kmeans,numstates)
    HMM_output5 = HMM(X_train5,numsymbols,kmeans,numstates)
    HMM_output6 = HMM(X_train6,numsymbols,kmeans,numstates)
    HMM_output8 = HMM(X_train8,numsymbols,kmeans,numstates)
    HMM_outputz = HMM(X_trainz,numsymbols,kmeans,numstates)
    
    C_dev_found = []
    total_len = len(C_dev)
    matched_len = 0
    scores = []
    i=0
    for devpoint in X_dev:
        symbolseq_devpoint = kmeans.predict(devpoint)
        score1 = forwardalgo(symbolseq_devpoint,HMM_output3,numstates)
        score2 = forwardalgo(symbolseq_devpoint,HMM_output5,numstates)
        score3 = forwardalgo(symbolseq_devpoint,HMM_output6,numstates)
        score4 = forwardalgo(symbolseq_devpoint,HMM_output8,numstates)
        score5 = forwardalgo(symbolseq_devpoint,HMM_outputz,numstates)
        score_i = [score1,score2,score3,score4,score5]
        scores.append(score_i)
        predicted_class = score_i.index(max(score_i))
        C_dev_found.append(predicted_class)
        if predicted_class == C_dev[i]:
            matched_len+=1
        i+=1
            
    
	
    scores = np.array(scores)
    print("Accuracy :-"+str((matched_len/total_len)*100))
    return (scores,C_dev_found)

n = len(C_dev)
(scores,C_dev_found1) = tempfun(15,3)
(fpr1,tpr1)=plot_ROC_curve(scores,n,"k=3")
(scores,C_dev_found2) = tempfun(15,4)
(fpr2,tpr2)=plot_ROC_curve(scores,n,"k=4")
(scores,C_dev_found3) = tempfun(15,5)
(fpr3,tpr3)=plot_ROC_curve(scores,n,"k=5")
(scores,C_dev_found4) = tempfun(15,6)
(fpr4,tpr4)=plot_ROC_curve(scores,n,"k=6")
(scores,C_dev_found5) = tempfun(15,7)
(fpr5,tpr5)=plot_ROC_curve(scores,n,"k=7")
plt.legend()
plt.show()


fig, ax_det = plt.subplots(1,1)
d1=DetCurveDisplay(fpr=fpr1,fnr=1-tpr1,estimator_name="DTW3").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr2,fnr=1-tpr2,estimator_name="DTW4").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr3,fnr=1-tpr3,estimator_name="DTW5").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr4,fnr=1-tpr4,estimator_name="DTW6").plot(ax=ax_det)
d1=DetCurveDisplay(fpr=fpr5,fnr=1-tpr5,estimator_name="DTW7").plot(ax=ax_det)
plt.legend()
plt.show()



draw_confusion(C_dev_found1)
plt.show()
draw_confusion(C_dev_found2)
plt.show()
draw_confusion(C_dev_found3)
plt.show()
draw_confusion(C_dev_found4)
plt.show()
draw_confusion(C_dev_found5)
plt.show()



