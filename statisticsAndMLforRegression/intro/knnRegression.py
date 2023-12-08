import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Load the inbuilt Boston data set of housing prices
data = datasets.load_boston()  

# Define predictors with pre-set feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Define target (housing prices) in another DataFrame with column name 'MEDV
Y = pd.DataFrame(data.target, columns=['MEDV'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=25)

# Instantiate KNN model with k = 3
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# predict response
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)



