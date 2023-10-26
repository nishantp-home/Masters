import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

N = 200
X = np.linspace(0, 10, N).reshape(N, 1)
y = np.sin(X)

Ntrain = 20
idx = np.random.choice(N, Ntrain)
Xtrain = X[idx]
ytrain = y[idx]

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(Xtrain, ytrain)
yknn = knn.predict(X)


dt = DecisionTreeRegressor()
dt.fit(Xtrain, ytrain)
ydt = dt.predict(X)

plt.scatter(Xtrain, ytrain)
plt.plot(X, y)
plt.plot(X, yknn, label='KNN')
plt.plot(X, ydt, label = 'Decision trees')
plt.legend()
plt.show()

