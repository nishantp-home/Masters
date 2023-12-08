import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Load the inbuilt Boston data set of housing prices
data = datasets.load_boston()  

# Define predictors with pre-set feature names
X = pd.DataFrame(data.data, columns=data.feature_names)

# Define target (housing prices) in another DataFrame with column name 'MEDV
Y = pd.DataFrame(data.target, columns=['MEDV'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=25)

# Instantiate the GB regressor
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)

importances = gbr.feature_importances_