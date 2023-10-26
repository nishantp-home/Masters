import numpy as np
from util import getData
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from future.utils import iteritems

class NaiveBayes(object):

    def fit(self, X, y, smoothing=10e-3):
        self.gaussians = dict()   # empty dictionary
        self.priors = dict()
        labels = set(y)       #created as set from a tuple y
        for c in labels:
            Xc = X[y == c]    #current X corresponding to class c
            self.gaussians[c] = {
                'mean': Xc.mean(axis=0),
                'var': Xc.var(axis=0)+smoothing,
            }
            self.priors[c] = float(len(y[y==c])) / len(y)

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)
    
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


class Bayes(NaiveBayes):
        
    def fit(self, X, y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict()   # empty dictionary
        self.priors = dict()
        labels = set(y)       #created as set from a tuple y
        for c in labels:
            Xc = X[y == c]    #current X corresponding to class c
            self.gaussians[c] = {
                'mean': Xc.mean(axis=0),
                'cov': np.cov(Xc.T) + np.eye(D) * smoothing,
            }
            self.priors[c] = float(len(y[y==c])) / len(y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__=='__main__':
    X, y = getData(10000)
    TrainCount =int(len(y) / 2)
    XTrain, yTrain = X[:TrainCount], y[:TrainCount]
    XTest, yTest = X[TrainCount:], y[TrainCount:]

    model = Bayes() #NaiveBayes()
    t0 = datetime.now()
    model.fit(XTrain, yTrain)
    print("Training time", (datetime.now()-t0))

    t0 = datetime.now()
    print("Train accuracy", model.score(XTrain, yTrain))
    print("Time to compute train accuracy", (datetime.now()-t0))
    
    t0 = datetime.now()
    print("Train accuracy", model.score(XTest, yTest))
    print("Time to compute test accuracy", (datetime.now()-t0))    








