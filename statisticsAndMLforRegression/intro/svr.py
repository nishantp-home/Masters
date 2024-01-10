# Support Vector Regression (SVR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load data from inbuilt dataset Boston housing prices
data = datasets.load_boston() # type: ignore
X = pd.DataFrame(data=data.data, columns=data.feature_names)   # Predictor
Y = pd.DataFrame(data=data.target, columns=['MEDV'])    # Target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=25)

# Instantiate SVR model
model = SVR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)