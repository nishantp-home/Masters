import numpy as np
from softKmeans import plot_k_means

def donut():

    N = 1000   # Sample count, dimensionality of samples = 2

    R_inner = 5.     # Inner radius
    R_outer = 10.    # Outer radius

    sampleCountPerCluster = int(N/2)  

    R1 = np.random.randn(sampleCountPerCluster) + R_inner
    theta = 2*np.pi*np.random.random(sampleCountPerCluster)
    X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

    R2 = np.random.randn(sampleCountPerCluster) + R_outer
    theta = 2*np.pi*np.random.random(sampleCountPerCluster)
    X_outer = np.concatenate([[R2*np.cos(theta)], [R2*np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])

    return X
 

def main():

# Cases where K-means fail
    
    # Example 1 : Donut problem
    X = donut()
    plot_k_means(X, 2, clusterEvolution=True)

    # Example 2 : Elongated Gaussians
    X = np.zeros((1000,2))
    X[:500,:] = np.random.multivariate_normal(mean=[0,0], cov=[[1,0],[0,20]], size=500)
    X[500:,:] = np.random.multivariate_normal(mean=[5,0], cov=[[1,0],[0,20]], size=500)
    plot_k_means(X, 2)

    # Example 3 : Gaussians of different densities
    X = np.zeros((1000, 2))
    X[:950,:] = np.array([0,0]) + np.random.randn(950,2)
    X[950:,:] = np.array([3,0]) + np.random.randn(50,2)
    plot_k_means(X,2)
    


if __name__ == '__main__':
    main()