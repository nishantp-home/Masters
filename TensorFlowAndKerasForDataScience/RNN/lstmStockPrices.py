# LSTM on Stock data
# Build a model that predicts the stock prices of a company 
# based on the prices of the previous few days.

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section12"
filePath += "\\all_stocks_5yr.csv"
df = pd.read_csv(filePath)

companies = df.Name.unique()

# get closing values of ZTS
z = df.loc[df['Name'] == 'ZTS']
z.info()

# Create an array with closing prices
trainingd = z.iloc[:, 4:5].values

# Normalizing vlaues 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(trainingd)

# x_train stores the values of closing prices of past 45 (or specified in timestamp) days
# y_train stores the values of closing prices of the present day
x_train = []
y_train = []
timestamp = 45
length = len(trainingd)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Prepare the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=120, return_sequences=True)) # 120 neurons in the hidden layer
# return_sequences=True makes LSTM layer to return the full history including outputs at all times
model.add(Dropout(0.2))

model.add(LSTM(units=120, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=120, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=120, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # output
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(x_train, y_train, epochs=25, batch_size=32)

# Forecasting (on other companies)
test_set = df.loc[df['Name']=='BA']   # change CBS to whatever company from the list
test_set = test_set.loc[:, test_set.columns == 'close']

# Storing the actual stock prices in y_test starting from 45th day as the previous 45 days
# are used to predict the present day value
y_test = test_set.iloc[timestamp:, 0].values

# Storing all values in a variable for generating an input array for our model
closing_price = test_set.iloc[:, 0:].values
closing_price_scaled = sc.transform(closing_price)

# The model will predict the values on x_test
x_test = []
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(closing_price_scaled[i-timestamp:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predicting the stock price values
y_pred = model.predict(x_test)
predicted_price = sc.inverse_transform(y_pred)

# Plotting the results
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predicted_price, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()