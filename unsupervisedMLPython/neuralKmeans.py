import numpy as np
import matplotlib.pyplot as plt
from softKmeans import get_simple_data
from sklearn.preprocessing import StandardScaler

# get the data and stadardize (scale) it #
X = get_simple_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# get shapes
N, D = X.shape 
K = 3

# initialize Neural network parameters
W = np.random.randn(D, K)

# Set hyperparameters
epochCount = 100
learningRate = 0.001
losses = []

# training loop
for i in range(epochCount):
    loss = 0
    for j in range(N):
        h = W.T.dot(X[j])   # K-length vector
        k = np.argmax(h)    # winning neuron / class

        # accumulate loss
        loss += (W[:, k] - X[j]).dot(W[:, k] - X[j])

        # Weight update
        gradient = (X[j] - W[:, k]) 
        W[:, k] += learningRate*gradient

    losses.append(loss)

plt.plot(losses)
plt.show()


# Show cluster assignments
H = np.argmax(X.dot(W), axis=1)
plt.scatter(X[:, 0], X[:, 1], c=H, alpha=0.5)
plt.show()
