import requests     #for making http requests
import numpy as np
import matplotlib.pyplot as plt
from util import getData

# Not working at the moment

X, y = getData()
N = len(y)

while True:
    i = np.random.choice(N)
    r = requests.post("http://localhost:8888/predict", data={'input': X[i]})
    j = r.json()
    print(str(j))
    print("target:", y[i])

    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.show()

    response = input("Continue? (Y/n)\n")
    if response in ('n', 'N'):
        break



