import numpy as np
import matplotlib.pyplot as plt
from softKmeans import get_simple_data
from scipy.cluster.hierarchy import dendrogram, linkage


def main():
    X = get_simple_data()

    Z = linkage(X, 'ward')
    print("Z.shape =", Z.shape)

    plt.title("Ward")
    dendrogram(Z)
    plt.show()


    Z = linkage(X, 'single')
    plt.title("Single")
    dendrogram(Z)
    plt.show()


    Z = linkage(X, 'complete')
    plt.title("Complete")
    dendrogram(Z)
    plt.show()






if __name__ == '__main__':
    main()