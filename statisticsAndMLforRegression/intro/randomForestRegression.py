import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

# Load the inbuilt Boston data set of housing prices
data = datasets.load_boston()  

# Define predictors with pre-set feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Define target (housing prices) in another DataFrame with column name 'MEDV
Y = pd.DataFrame(data.target, columns=['MEDV'])

# Split dataset into train-test data sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25)
y_train, y_test = y_train.values.reshape(len(y_train),), y_test.values.reshape(len(y_test),)

# Random forest regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score


# Accuracy of prediction
MSE = mean_squared_error(y_test, y_pred=y_pred)
r2Score = r2_score(y_test, y_pred)






# Check which features are important
importances = regr.feature_importances_    #RF based predictor variable importance
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Dataset with important features
X1 = X[['RM', 'LSTAT']]

# Split data set
X1_train, X1_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=25)

regr.fit(X1_train, y_train)
y1_pred  = regr.predict(X1_test)
MSE = mean_squared_error(y_test, y1_pred)
importances = regr.feature_importances_
print(MSE, importances)