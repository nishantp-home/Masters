import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from datasets import (
    circles,
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances
)

#Generating data-set
#X = circles()
#X = moons()
#X = blobs()
#X = anisotropic()
#X = random()
X = varied_variances()

dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan.fit(X)

#get inliers and their cluster
X_inlier = X[dbscan.labels_ != -1]
y_inlier = dbscan.labels_[dbscan.labels_ !=-1]

#get outliers
X_outlier = X[dbscan.labels_ == -1]

#plot scattered data
plt.figure()
plt.scatter(X_inlier[:,0], X_inlier[:,1],c=y_inlier,cmap='Dark2', alpha=0.5)
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='k', alpha=0.5)

plt.show()