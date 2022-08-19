class KNN :
    def __init__(self,train_data,test_data,train_class,test_class,classes,topK):
        self.train_data = train_data
        self.train_class = train_class
        self.test_data = test_data
        self.test_class = test_class
        self.data_len = len(train_data)
        self.dist = []
        self.cls = []
        self.topK = topK
        self.classes = classes

    def compute_distance(self):
        for p in self.test_data:
            for_point = []
            for t in self.train_data:
                sum = 0
                for i in range(len(self.train_data[0])):
                   sum += (t[i] - p[i]) ** 2
                for_point.append(sum)
            zipp = sorted(zip(for_point,self.train_class))
            dist, cls = zip(*zipp)
            self.dist.append(dist)
            self.cls.append(cls)


    def classify(self):
        classified = []
        scores = []
        for i in range(len(self.test_data)):
            #print(f"point {i}")
            cls = [0] * (self.classes + 1)
            for j in range(self.topK):
                cls[self.cls[i][j]] += 1
            for i in range(self.classes + 1):
                cls[i]=cls[i]/self.topK
            classified.append(cls.index(max(cls)))
            scores.append(cls[1:])
        return classified,scores
