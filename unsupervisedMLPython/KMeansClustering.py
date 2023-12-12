import numpy as np
import matplotlib.pyplot as plt


# Generate synthetic dataset -----------------------------
N = 300     # Sample count
D = 2       # Feature count
K = 3       # cluster identity count
X = np.zeros((N, D))
X[0:100,:] = np.random.randn(100, D) + np.array((0,0))
X[100:200,:] = np.random.randn(100, D) + np.array((5,5))
X[200:300,:] = np.random.randn(100, D) + np.array((0,5))

plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c='b', alpha=0.7)
plt.title('Unclustered 2-D data')
# plt.scatter(Xmean[:,0], Xmean[:, 1], c='r', s=100, marker='*')
plt.show()

clusterIndentityVector = np.zeros(N)
oldClusterIdentityVector = np.zeros(N)

# Clustering algorithm -----------------------------------
# randomly initialize 'K' means from X
mean = X[np.random.choice(N, size=K, replace=False),:]    

# Find clusters iteratively until no change/ convergence
iter = 0
costs = []
savedClusterIndentities = []
minimum_distances = np.zeros(N)
while True:
    oldClusterIdentityVector = clusterIndentityVector.copy()
    savedClusterIndentities.append(oldClusterIdentityVector)
    iter += 1
    print("Iteration:", iter)
    for n in range(N):
        minimum_distance = float('inf')
        closest_k = -1
        for k in range(K):
            distance = (X[n]-mean[k]).dot(X[n]-mean[k])
            if distance < minimum_distance:
                minimum_distance = distance
                closest_k = k
        minimum_distances[n] = minimum_distance
        clusterIndentityVector[n] = closest_k

    cost = minimum_distances.sum()
    costs.append(cost)

# Compute means
    for k in range(K):
        mean[k] = X[clusterIndentityVector==k].mean(axis=0)
    
    if ((clusterIndentityVector==oldClusterIdentityVector).all()):
        print("Loop over")
        break

plt.figure('Final clusters')
plt.scatter(X[:,0], X[:,1], c=clusterIndentityVector, alpha=0.7)
plt.scatter(mean[:,0], mean[:,1], c='red', marker='*')
plt.title('Final clusters (with means)')
plt.show()

plt.figure('Cost with each iteration')
plt.plot(costs, marker='x')
plt.title('Cost Vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()


# show clustering progress
M = len(savedClusterIndentities)
figColumns = 2
figRows = int(M/figColumns) if (M%figColumns == 0) else int((M+1)/figColumns)
fig, ax = plt.subplots(figRows, figColumns, sharex=True, sharey=True, figsize=(10, 20))
for i in range(M):
    clusterIndentityVector = savedClusterIndentities[i]
    rowIndex = i // figColumns
    columnIndex = i % figColumns
    ax[rowIndex, columnIndex].scatter(X[:,0], X[:,1], c=clusterIndentityVector)
    ax[rowIndex, columnIndex].set_title('Clusters after %d iterations' % (i))



