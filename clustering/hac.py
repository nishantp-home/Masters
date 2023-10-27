import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from datasets import (
    circles,
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances
)

# Generating data-set
# X = circles()
# X = moons()
# X = blobs()
# X = anisotropic()
# X = random()
X = varied_variances()


hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
hac.fit(X)

plt.scatter(X[:,0], X[:,1], c=hac.labels_, alpha=0.5)
plt.show()