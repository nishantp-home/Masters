import numpy as np
import matplotlib.pyplot as plt

def d(u, v):
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost+=R[n,k]*d(M[k], X[n])
    return cost

def computeMeans(R, X):
    K = R.shape[1]
    D = X.shape[1]
    Means = np.zeros((K, D))    # M: Array containing 'K' means of dimensionality 'D'
    for k in range(K):
        Means[k] = R[:, k].dot(X) / R[:,k].sum()
    return Means



def plotClusterEvolution(X, K, List_of_R, plotSize=(3,2), ListOfKmeans=None):
    """This function plots the evolution of clusters with iterations in one figure"""

    figureSize = tuple(plotSize)
    iterationCount = len(List_of_R)
    figureCount=figureSize[0]*figureSize[1]

    print('iterationCount:', iterationCount)
    print('figureCount:', figureCount)

    if iterationCount <= figureCount:
        print('Number of iterations are less than number of requested subplots')
        print('Rearranging figures into figSize=(1 x 2)')
        figureSize = (1,2)
        figureCount = figureSize[0]*figureSize[1]


    fig, ax = plt.subplots(figureSize[0], figureSize[1], sharex=True, sharey=True, layout="constrained", subplot_kw=dict(box_aspect=1))
    random_colors = np.random.random((K, 3))
    step, remainder = iterationCount // (figureCount-1),  iterationCount % (figureCount-1)


    for figureIndex in range(figureCount):

        iteration = figureIndex*step
        # Step larger than total iteration count, store the last iteration plot
        if(iteration >= iterationCount-1):   
            iteration = iterationCount-1
            figureIndex = figureCount-1

        if (figureIndex == figureCount - 1) and (remainder > 0):   #store the last iteration plot in last subplot
            iteration = iterationCount-1

        R = List_of_R[iteration].copy()
        color = R.dot(random_colors)

        if figureSize[0] == 1 or figureSize[1] == 1:    # if n_row=1 or n_column=1 for ax, ax is a vector
            ax[figureIndex].scatter(X[:,0], X[:,1], c=color, alpha=0.5)
            if ListOfKmeans != None:
                Kmeans = ListOfKmeans[iteration]
                ax[figureIndex].scatter(Kmeans[:,0], Kmeans[:,1], c='red', marker='*')
            ax[figureIndex].set_title('after %d iterations' % (iteration))
        else:
            rowIndex, columnIndex = figureIndex // figureSize[1], figureIndex % figureSize[1]
            ax[rowIndex, columnIndex].scatter(X[:,0], X[:,1], c=color, alpha=0.5)
            if ListOfKmeans != None:
                Kmeans = ListOfKmeans[iteration]
                ax[rowIndex, columnIndex].scatter(Kmeans[:,0], Kmeans[:,1], c='red', marker='*')
            ax[rowIndex, columnIndex].set_title('after %d iterations' % (iteration))
    
    fig.suptitle("Evolution of Cluster with iteration", fontsize=16)
    plt.show() 




def plotCostIter(costs):
    plt.plot(costs)
    plt.title("Cost with iteration")
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.show()


def plotCluster(X, K, R, Means=None):
    # generate K random rgb colors (each color: vector of size 3) 
    random_colors = np.random.random((K, 3))    
    # colormap for each sample based on computed final Responsibility matrix
    colors = R.dot(random_colors)               
    plt.scatter(X[:,0], X[:,1], c=colors)
    if Means.any() != None:
        plt.scatter(Means[:,0], Means[:,1], c='red', marker='*')
    plt.title('Final Cluster Plot')
    plt.show()


def plot_k_means(X, K, max_iter=20, beta=1.0, tolerance=0.1, clusterEvolution=False):

    N, D = X.shape          # N: Sample count, D: Dimensionality
    Means = np.zeros((K, D))    # M: Array containing 'K' means of dimensionality 'D'
    R = np.zeros((N, K))    # R: Responsibility matrix

    # Randomly initialize K means from the dataset X
    for k in range(K):
        Means[k] = X[np.random.choice(N)]

    costs = []   # List for storing cost at each iteration
    List_of_R = [] # Append to list of R
    List_of_Means = []

    # Soft K-means clustering algorithm
    for i in range(max_iter):
        
        # Store the computed K means at each iteration
        List_of_Means.append(Means.copy())
        List_of_R.append(R.copy())
        
        for k in range(K):
            for n in range(N):
                # Compute responsibility matrix 
                # R[n,k] = np.exp(-beta*d(Means[k], X[n])) / np.sum(
                #     np.exp(-beta*d(Means[j], X[n])) for j in range(K))    
                iterable = (np.exp(-beta*d(Means[j], X[n])) for j in range(K))
                R[n,k] = np.exp(-beta*d(Means[k], X[n])) / np.sum(np.fromiter(iterable, dtype=np.float64))    
                 
        # Recompute means from the computed responsibility matrix R and dataset X
        Means = computeMeans(R, X)

        # Compute cost at each iteration, 
        # and append it to the list for plotting
        costs.append(cost(X, R, Means))

        # Convergence criterion
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < tolerance:
                break

# Plot relevant results
    plotCostIter(costs=costs)
    plotCluster(X, K, R, Means)
    if clusterEvolution == True:
        plotClusterEvolution(X, K, List_of_R, plotSize=(2,2), ListOfKmeans=List_of_Means)


## Same as plot_k_means, but returns M and R 
def plot_k_means2(X, K, max_iter=20, beta=1.0, tolerance=0.1, showPlots=True, showClusterEvolution=False):

    N, D = X.shape              # N: Sample count, D: Dimensionality
    Means = np.zeros((K, D))    # M: Array containing 'K' means of dimensionality 'D'
    R = np.zeros((N, K))    # R: Responsibility matrix

    # Randomly initialize K means from the dataset X
    for k in range(K):
        Means[k] = X[np.random.choice(N)]

    costs = []   # List for storing cost at each iteration
    List_of_R = [] # Append to list of R
    List_of_Means = []

    # Soft K-means clustering algorithm
    for i in range(max_iter):
        
        # Store the computed K means at each iteration
        List_of_Means.append(Means.copy())
        List_of_R.append(R.copy())
        
        for k in range(K):
            for n in range(N):
                # Compute responsibility matrix 
                # R[n,k] = np.exp(-beta*d(Means[k], X[n])) / np.sum(
                #     np.exp(-beta*d(Means[j], X[n])) for j in range(K))    
                iterable = (np.exp(-beta*d(Means[j], X[n])) for j in range(K))
                R[n,k] = np.exp(-beta*d(Means[k], X[n])) / np.sum(np.fromiter(iterable, dtype=np.float64))    
                 
        # Recompute means from the computed responsibility matrix R and dataset X
        Means = computeMeans(R, X)

        # Compute cost at each iteration, 
        # and append it to the list for plotting
        costs.append(cost(X, R, Means))

        # Convergence criterion
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < tolerance:
                break

# Plot relevant results
    if showPlots == True:
        plotCostIter(costs=costs)
        plotCluster(X, K, R, Means)
    
    if showClusterEvolution == True:
        plotClusterEvolution(X, K, List_of_R, plotSize=(2,2), ListOfKmeans=List_of_Means)

    return Means, R

def get_simple_data():
    # assume 3 means
    D = 2 # so we can visualize it more easily
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900 # number of samples
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3
    return X


def main():

    X = get_simple_data()

    # Visualize data
    plt.scatter(X[:,0], X[:,1])
    plt.show()


    # Analyze soft K-means clustering for different hyperparameters (K, beta)
    # K = 3   
    # plot_k_means(X, K)

    K = 5   
    plot_k_means(X, K, max_iter=30, clusterEvolution=True)

    # K = 5
    # plot_k_means(X, K, max_iter=30, beta=0.6)


if __name__ == '__main__':
    main()
