import numpy as np
import matplotlib.pyplot as plt
from util import getData as getMNIST
from datetime import datetime

def getData():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    y = np.sign(X.dot(w) + b)

    return X, y

def getSimpleXOR():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0,1,1,0])
    return X, y


class Perceptron():
    def fit(self, X, y, learningRate=1.0, epochs=1000):
        D = X.shape[1]    #Dimensionality
        self.w = np.random.randn(D)
        self.b = 0

        N = len(y)
        costs = []

        for epoch in range(epochs):
            yHat = self.predict(X)
            incorrect = np.nonzero(y != yHat)[0]
            if len(incorrect) == 0:
                break

            i = np.random.choice(incorrect)   #choose random sample from incorrect samples
            self.w += learningRate*y[i]*X[i]
            self.b += learningRate*y[i]

            c = len(incorrect) / float(N)
            costs.append(c)

        print("final w:", self.w, "final b:", self.b, "epochs:", epoch+1, "/", epochs)
        plt.plot(costs)
        plt.show()


    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)
    
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)



if __name__ == '__main__':
    # X, y = getData()
    # plt.scatter(X[:,0], X[:,1], c=y, s=100, alpha=0.5)
    # plt.show()

    # Ntrain = int(len(y) / 2)
    # Xtrain, yTrain = X[:Ntrain], y[:Ntrain]
    # Xtest, yTest = X[Ntrain:], y[Ntrain:]

    # model = Perceptron()
    # t0 = datetime.now()
    # model.fit(Xtrain, yTrain)
    # print("Training time", (datetime.now() - t0))

    # t0 = datetime.now()
    # print("Training accuracy:", model.score(Xtrain, yTrain))
    # print("Time to compute training accuracy:", (datetime.now() - t0))

    # t0 = datetime.now()
    # print("Testing accuracy:", model.score(Xtest, yTest))
    # print("Time to compute testing accuracy:", (datetime.now() - t0))

    X, y = getMNIST()
    idx = np.logical_or(y == 0, y == 1)
    X = X[idx]
    y = y[idx]
    y[y==0] = -1
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X, y, learningRate=0.01)
    print("MNIST train accuracy:", model.score(X, y))

    print("")
    print("XOR results")
    X, y = getSimpleXOR()
    model.fit(X, y)
    print("XOR accuracy:", model.score(X, y))


