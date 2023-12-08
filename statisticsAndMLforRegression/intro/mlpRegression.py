# Multilayer Perceptron (MLP)  for Regression

import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


# Load the inbuilt Boston data set of housing prices
data = datasets.load_boston()  

# Define predictors with pre-set feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Define target (housing prices) in another DataFrame with column name 'MEDV
Y = pd.DataFrame(data.target, columns=['MEDV'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=25)

# Instantiate the MLP regressor
mlr = MLPRegressor(activation='logistic', 
                   solver='lbfgs', 
                   alpha=0.0001, random_state=1)
mlr.fit(X_train,y_train)
y_pred= mlr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)

# Modify the hidden layer
clf = MLPRegressor(hidden_layer_sizes=(15,),
                    max_iter=100000)
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)
r2score2 = r2_score(y_test, y_pred2)