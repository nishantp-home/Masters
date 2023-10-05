import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN

def getData():
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            y[n] = t
            n = n + 1
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return X, y

if __name__ == '__main__':
    X, y = getData()
    print("X=", X)
    print("y=", y)

    plt.scatter(X[:, 0], X[:, 1], s=500, c=y, alpha=1.0)
    plt.show()
    
    model = KNN(3)
    model.fit(X,y)
    print("Train accuracy =", model.score(X,y))