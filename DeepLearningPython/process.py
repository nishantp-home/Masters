import pandas as pd
import numpy as np


def get_data():
    fileName = "ecommerce_data.csv"
    df = pd.read_csv(fileName)
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalize column 2: n_products_viewed and 3: visit_duration
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3))   # Three extra classes

    X2[:, 0:(D-1)] = X[:, 0:(D-1)]   # First 5 columns are same as X

    for n in range(N):
        t = int(X[n, D-1])  #time of day
        X2[n, t+D-1] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    return X2, Y


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2





    
