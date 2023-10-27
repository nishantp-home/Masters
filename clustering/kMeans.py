import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
# X = blobs()
# X = anisotropic()
# X = random()
X = varied_variances()

#Instantiating kmeans and running (using fit function) the clustering algorithm
kMeans = KMeans(n_clusters=3, random_state=17)
kMeans.fit(X)

#plot scattered data
plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.5)

#plotting clustered scattered data
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kMeans.labels_, alpha=0.5)

plt.show()