import numpy as np
from util import getData, getXOR, getDonut
from datetime import datetime

def entropy(y):
    N = len(y)
    s1 = (y==1).sum()
    if 0 == s1 or N ==s1:
        return 0
    
    p1 = float(s1) / N
    p0 = 1 - p1

    return -p0*np.log2(p0) - p1*np.log2(p1)


class TreeNode:
    def __init__(self, depth=0, maxDepth=None):
        self.depth = depth
        self.maxDepth = maxDepth

    def fit(self, X, y):
        if len(y) == 1 or len(set(y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = y[0]
        else:
            D = X.shape[1]
            cols = range(D)

            maxIg = 0
            bestCol = None
            bestSplit = None
            for col in cols:
                ig, split = self.findSplit(X, y, col)
                if ig > maxIg:
                    maxIg = ig
                    bestCol = col
                    bestSplit = split

            if maxIg == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(y.mean())
            else:
                self.col = bestCol
                self.split = bestSplit

                if self.depth == self.maxDepth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(y[X[:, bestCol] < self.split].mean()),
                        np.round(y[X[:, bestCol] >= self.split].mean())
                    ]

                else:
                    leftIdx = (X[:, bestCol] > bestSplit)
                    Xleft = X[leftIdx]
                    Yleft = y[leftIdx]
                    self.left = TreeNode(self.depth + 1, self.maxDepth)
                    self.left.fit(Xleft, Yleft)

                    rightIdx = (X[:, bestCol] >= bestSplit)
                    Xright = X[rightIdx]
                    Yright = y[rightIdx]
                    self.right = TreeNode(self.depth + 1, self.maxDepth)
                    self.right.fit(Xright, Yright)


    def findSplit(self, X, y, col):
        xValues = X[:, col] 
        sortIdx = np.argsort(xValues)
        xValues = xValues[sortIdx]
        yValues = y[sortIdx]

        boundaries = np.nonzero(yValues[:-1] != yValues[1:])[0]
        bestSplit = None
        maxIg = 0
        for i in range(len(boundaries)):
            split = (xValues[i] + xValues[i+1]) / 2
            ig = self.informationGain(xValues, yValues, split)
            if ig > maxIg:
                bestSplit = split
            return maxIg, bestSplit
        
    def informationGain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x > split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)
    

    def predictOne(self, x):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predictOne(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predictOne(x)
                else:
                    p = self.prediction[1]

        else:
            p = self.prediction
        return p
    


    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predictOne(X[i])
        
        return P
    

class DecisionTree:
    def __init__(self, maxDepth=None):
        self.maxDepth = maxDepth


    def fit(self, X, y):
        self.root = TreeNode(maxDepth=self.maxDepth)
        self.root.fit(X, y)

    def predict(self, X):
        return self.root.predict(X)
    
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P==y)
    

if __name__ == '__main__':

    X, y = getData()
    idx = np.logical_or(y == 0, y == 1)
    X = X[idx]
    y = y[idx]


    Ntrain = int(len(y) / 2)
    Xtrain, yTrain = X[:Ntrain], y[:Ntrain]
    Xtest, yTest = X[Ntrain:], y[Ntrain:]

    model = DecisionTree()
    t0 = datetime.now()
    model.fit(Xtrain, yTrain)
    print("Training time", (datetime.now() - t0))

    t0 = datetime.now()
    print("Training accuracy:", model.score(Xtrain, yTrain))
    print("Time to compute training accuracy:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Testing accuracy:", model.score(Xtest, yTest))
    print("Time to compute testing accuracy:", (datetime.now() - t0))




    



