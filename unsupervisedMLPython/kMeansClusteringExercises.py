import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

#Ex1: Generate synthetic dataset X, cluster identity Y, compute the cluster means and visualize scattered plot

# Generate synthetic dataset
N = 300     # Sample count
D = 2       # Feature count
K = 3       # cluster identity

Ncluster = int(N/K)
X1 = np.random.randn(Ncluster, D) + np.array([0,0])
X1mean = X1.mean(axis=0)
X2 = np.random.randn(Ncluster, D) + np.array([5,5])
X2mean = X2.mean(axis=0)
X3 = np.random.randn(Ncluster, D) + np.array([0,5])
X3mean = X3.mean(axis=0)
X = np.vstack([X1,X2,X3])
Xmean = np.vstack([X1mean, X2mean, X3mean])

# Generate cluster identity
Y = np.array([0]*100 + [1]*100+[2]*100)
#Y = np.random.random_integers(low=0, high=K-1, size=(N,))

# Visualize the data
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.7)
plt.scatter(Xmean[:,0], Xmean[:, 1], c='r', s=100, marker='*')
plt.show()

# Ex2
# Given dataset X and cluster means, find cluster identities (Y1)

# initialize cluster identity vector Y1
Y1 = np.zeros(N).astype(int)

# Shuffle dataset X
np.random.shuffle(X)

# Compute distances of all Xs from cluster means, in vectorized form
D1 = np.sqrt(np.power(X - Xmean[0], 2).sum(axis=1))
D2 = np.sqrt(np.power(X - Xmean[1], 2).sum(axis=1))
D3 = np.sqrt(np.power(X - Xmean[2], 2).sum(axis=1))

# Compute Y1 in the vectorized form using numpy functions
ind0 = np.intersect1d(np.asarray(D1 < D2).nonzero(), np.asarray(D1 < D3).nonzero())
Y1[ind0] = 0
# np.asarray(condition).nonzero() return the array of indices with true values
# np.intersect1d returns the array of common entries of two arrays

ind1 = np.intersect1d(np.asarray(D2 < D1).nonzero(), np.asarray(D2 < D3).nonzero())
Y1[ind1] = 1

ind2 = np.intersect1d(np.asarray(D3 < D1).nonzero(), np.asarray(D3 < D2).nonzero())
Y1[ind2] = 2

# Visualize dataset with clusters
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=Y1, alpha=0.7)
plt.scatter(Xmean[:,0], Xmean[:, 1], c='r', s=100, marker='*')
plt.show()



