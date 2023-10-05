from myKNN import KNN
from util import get_xor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, y = get_xor()
    print("X = ", X)
    print("y = ", y)

    plt.scatter(X[:, 0], X[:, 1], s=100, c=y, alpha=0.5)
    plt.show()

    model = KNN(3)
    model.fit(X,y)
    print("Accuracy =", model.score(X, y))