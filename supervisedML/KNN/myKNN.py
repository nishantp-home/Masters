# from __future__ import print_function, division
# from future.utils import iteritems
# from builtins import range, input

import numpy as np
from sortedcontainers import SortedList
from util import getData
from datetime import datetime
from future.utils import iteritems
import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))
                
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1

            maxVotes = 0
            maxVotesClass = -1

            for v, count in iteritems(votes):
                if count > maxVotes:
                    maxVotes = count
                    maxVotesClass = v
                    
            y[i] = maxVotesClass
        
        return y
    
    def score(self, X, y):
        P = self.predict(X)

        return(np.mean(P == y))


if __name__ == '__main__':
    SampleCount = 2000
    X , y = getData(SampleCount)
    trainingSampleCount = 1000
    xTrain, yTrain = X[0:trainingSampleCount], y[0:trainingSampleCount]          
    xTest, yTest = X[trainingSampleCount:], y[trainingSampleCount:]  
    trainScores = []
    testScores = []

    ks = (1,2,3,4,5)
    for k in ks:
        print("\nk:", k)
        knn = KNN(k)
        knn.fit(xTrain, yTrain)
        
        t0 = datetime.now()
        trainScore = knn.score(xTrain, yTrain)
        trainScores.append(trainScore)
        print("Training accuracy =", trainScore)
        print("Computation time (training accuracy) =", datetime.now()- t0)
  
        t0 = datetime.now()
        testScore = knn.score(xTest, yTest)
        testScores.append(testScore)
        print("Test accuracy =", testScore)
        print("Computation time (testingg accuracy) =", datetime.now()- t0)


    plt.plot(ks, trainScores, label = "Training accuracy")
    plt.plot(ks, testScores, label = "Testing accuracy")
    plt.legend()
    plt.show()

