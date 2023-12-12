import numpy as np
import matplotlib.pyplot as plt
from softKmeans import plot_k_means2, cost, get_simple_data


# generate Cost Vs K plot for K = 1-9
def main():
    X = get_simple_data()

    # Visualize data
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    costs = np.empty(10)
    costs[0] = None
    for k in range(1,10):
        M, R = plot_k_means2(X, k, showPlots=False)
        c = cost(X, R, M)
        costs[k] = c

    plt.plot(costs)
    plt.title('Cost vs K')
    plt.show()


if __name__ == '__main__':
    main()


