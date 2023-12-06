import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    Z = 1/(1+ np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct+=1
        
    return float(n_correct) / n_total


def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]     # shape of Z = N * M

    ret1 = np.zeros((M, K))

    # slow method
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m, k] += (T[n, k] - Y[n, k]) * Z[n, m]

#   Fast method
    # ret2 = np.array((M, K))
    # for n in range(N):
    #     for k in range(K):
    #         ret2[:, k] += (T[n, k] - Y[n, k]) * Z[n, :]

# Faster method
    # ret3 = np.array((M, K))
    # for n in range(N):
    #     ret3 = np.outer(Z[n], (T[n]-Y[n]))
    # ret3.sum()

# Fastest method
    # derivative = Z.T.dot(T-Y)

    return Z.T.dot(T-Y)

def derivative_b2(T, Y):
    return (T-Y).sum(axis=0)    #sum all rows, result = one row vector


def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

# # slow
#     derivative = np.zeros((D, M))
#     for n in range(N):
#         for k in range(K):
#             for m in range(M):
#                 for d in range(D):
#                     derivative[d, m] += (T[n,k]-Y[n,k])* W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
    dZ = (T - Y).dot(W2.T) * Z* (1 - Z)

    return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
    return ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main():
    # Create data
    Nclass = 500
    D = 2    # Dimensionality of input
    M = 3    # Hidden layer size
    K = 3    # Number of classes

    # Generate three gaussian clouds X1, X2, X3
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])   # gaussian cloud centered at (0,-2)
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])    # Gaussian cloud centered at (2, 2)
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])    # Gaussian cloud centered at (2, 2)
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)

    T = np.zeros((N, K))
    for i in range(N):
        T[i][Y[i]] = 1

    # Lets see how the data looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

# randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    epochCount = 100000

    costs = []
    for epoch in range(epochCount):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification rate:", r)
            costs.append(c)

        gW2 = derivative_w2(hidden, T, output)
        gb2 = derivative_b2(T, output)
        gW1 = derivative_w1(X, hidden, T, output, W2)
        gb1 = derivative_b1(T, output, W2, hidden)

        W2 += learning_rate * gW2
        b2 += learning_rate * gb2
        W1 += learning_rate * gW1
        b1 += learning_rate * gb1
    
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
