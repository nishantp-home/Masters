import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data


def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1

    return ind


X, Y = get_data()       # get nomralized data from csv file
X, Y = shuffle(X, Y)    # shuffle the data
Y = Y.astype(np.int32)   
D = X.shape[1]          # Dimensionality of input
K = len(set(Y))         # Number of classes

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W = np.random.randn(D, K)   # Gaussian distributed values of weights in W matrix
b = np.random.randn(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return - np.mean(T*np.log(pY))

train_costs = []
test_costs = []
learning_rate = 0.001
epochCount = 10000
for i in range(epochCount):
    pYtrain = forward(Xtrain, W, b)
    pytest = forward(Xtest, W, b)
    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pytest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate* Xtrain.T.dot(pYtrain - Ytrain_ind)
    b -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)
    if i % 1000 == 0:
        print("iteration:", i, "training cost:", ctrain, "testing cost:", ctest)

print("Final train classification rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification rate:", classification_rate(Ytest, predict(pytest)))


plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()
