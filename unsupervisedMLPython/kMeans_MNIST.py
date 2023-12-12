# K-Means clustering analysis of MNIST data
# MNIST dataset: Image data, each image is a D = 28*28 = 784 dimensional vector
# There are N = 42000 samples
# You can plot the image by reshaping to (28,28) and using plt.imshow()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from softKmeans import plot_k_means2
from datetime import datetime
import tensorflow as tf
from sklearn import datasets



def get_MNIST_Data(limit=None):
    mnist = tf.keras.datasets.mnist.load_data()

    X, y = mnist[0][0], mnist[0][1]

    # Reshape X from 60000*28*28 to 60000*784
    N = X.shape[0]
    D = X.shape[1]*X.shape[2]
    X = X.reshape(N, D)  # Samples

    # Scale from 0-255 to 0-1
    X = X/255

    if limit is not None:
        X, y = X[:limit], y[:limit]
    
    return X, y


def purity(Y, R):
    # Maximum purity is 1, higher is better
    N, K = R.shape
    p = 0    #purity value p, initialize to zero
    for k in range(K):
        max_intersection = 0
        for j in range(K):
            intersection = R[Y==j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
        p += max_intersection
    return p / N


# Hard labels
def purity2(Y, R):
    C = np.argmax(R, axis=1) # cluster assignment
    N = len(Y)  # sample count
    K = len(set(Y))  # label count

    total = 0.0
    for k in range(K):
        max_intersection = 0
        for j in range(K):
            intersection = ((C==k) & (Y==j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N





def DBI(X, M, R):
    # Davies-Bouldin Index
    # ratio b/w sum of std deviations b/w 2 clusters / distance b/w cluster means
    # Lower is better
    N, D = X.shape
    K = R.shape[1]

    # get sigmas, i.e., Mean sample distance for each cluster 
    sigma= np.zeros(K)
    for k in range(K):
        diff = X - M[k]   # Shape NxD
        squareDistances = (diff*diff).sum(axis=1)
        weightedSquaredDistances = R[:, k]*squareDistances
        sigma[k] = np.sqrt((weightedSquaredDistances.sum() / R[:, k].sum()))

    # Calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio    

        dbi += max_ratio

    dbi /= K

    return dbi


# Hard labels
def DBI2(X, R):
    N, D = X.shape
    K = R.shape[1]

    # get sigmas and means 
    sigma= np.zeros(K)
    M = np.zeros((K, D))
    assignments = np.argmax(R, axis=1)
    for k in range(K):
        Xk = X[assignments==k]
        M[k] = Xk.mean(axis=0)
        n = len(Xk)
        diffs = Xk - M[k]
        sq_diffs = diffs*diffs
        sigma[k] = np.sqrt(sq_diffs.sum() / n)

    # calculate DB index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
            
        dbi += max_ratio

    dbi /= K

    return dbi



def main():
    # mnist data
    X, Y = get_MNIST_Data(10000)

    print("Number of data points:", len(Y))
    M, R = plot_k_means2(X, len(set(Y)))

    # Exercise: Try different values of K and compare the evaluation metrics
    print("Purity:", purity(Y, R))
    print("Purity 2 (hard clusters):", purity2(Y, R))
    print("DBI:", DBI(X, M, R))
    print("DBI 2 (hard clusters):", DBI2(X, R))

    # plot the mean images
    # they should look like digits
    for k in range(len(M)):
        im = M[k].reshape(28, 28)
        plt.imshow(im, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main()
