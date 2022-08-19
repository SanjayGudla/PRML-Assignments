import numpy as np
import seaborn as sn
import pandas as pd 
class plots :
    def __init__(self,scores,n,s):
        self.scores = scores
        self.n = n
        self.s = s

    def plot_ROC_curve(scores, n,num_classes,C_dev):
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
                for j in range(num_classes):
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
        return (fpr,tpr)


    def draw_confusion(res1,num_classes,C_dev):
        conf= np.array([[0 for j in range(num_classes)] for i in range(num_classes)])
        for i in range(np.size(res1)):
            conf[C_dev[i]-1][res1[i]-1] += 1
        df_cm = pd.DataFrame(conf)
        as1 = sn.heatmap(df_cm, annot=True,fmt=".1f")
        as1.set_xlabel('true class')
        as1.set_ylabel('predicted class')
