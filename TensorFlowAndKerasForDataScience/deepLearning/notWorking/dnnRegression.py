import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics, preprocessing

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Access the Boston housing data
boston = load_boston()

# Convert data to date frame
bostonDF = pd.DataFrame(boston.data)
bostonDF.columns = boston['feature_names']  # Assign column names
 
X = bostonDF  # Predictors
y = boston.target   # Response

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=101)

# Scale the data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

# DNN Regressor
# Build a 3 layer fully connected DNN
#feature_column = boston.feature_names
feature_column = {k: tf.feature_column.numeric_column(k)  for k in boston.feature_names}
regressor = tf.estimator.DNNRegressor(feature_columns=feature_column, hidden_units=[10, 20, 10])

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={'variable': X_train} , y=y_train, num_epochs = None, shuffle=True, batch_size=1)

regressor.train(input_fn=train_input_fn, steps=5000)

y_predicted = regressor.predict(scaler.transform(X_test))
score = metrics.mean_squared_error(y_predicted, y_test)
print('MSE: {0:f}'.format(score))


