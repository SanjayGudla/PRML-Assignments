import math

import numpy as np


class Logistic:
    def __init__(self,train,dim,cls,test,train_cls):
        self.train = train
        self.train_cls = train_cls
        self.dim = dim
        self.cls = cls
        self.test = test
        self.w_k = np.zeros((cls,dim))
        self.probs = []
        self  .estimate_w()
        #return self.classify()

    def add_1(self):
        train = []
        for point in self.train:
            l = [1]
            l.extend(point)
            train.append(np.array(l))
        test = []
        for point in self.test:
            l = [1]
            l.extend(point)
            test.append(np.array(l))

        self.train = train
        self.test = test


    def estimate_probs(self,point):
        a_ks = []
        #print(np.shape(point))
        #print(np.shape(self.w_k[0]))
        for i in range(self.cls):
            a_ks.append(np.dot(self.w_k[i].T,point))
        #print(a_ks)
        ex = [math.exp(x) for x in a_ks]

        self.probs.append([x/sum(ex) for x in ex])

        return ([x/sum(ex) for x in ex])


    def estimate_w(self):
        learn = 10 ** (-3)
        iterations = 300

        #self.add_1()

        for _ in range(iterations):
            self.probs = []
            for point in self.train:
                self.estimate_probs(point)

            w_j = np.zeros((self.cls, self.dim))

            for j,point in enumerate(self.train):
                cls = self.train_cls[j] - 1
                for i in range(self.cls):
                    if cls == i:

                        w_j[i] += point * (self.probs[j][i] - 1)

                    else :

                        w_j[i] += point * (self.probs[j][i])


            #print(w_j)
            self.w_k = self.w_k - learn * w_j
            #print(self.w_k)

    def classify(self):
        classified = []
        #print(self.w_k)
        scores = []
        for point in self.test:
            prob = self.estimate_probs(point)
            scores.append(prob)
            classified.append(prob.index(max(prob)) + 1)

        return classified,scores
