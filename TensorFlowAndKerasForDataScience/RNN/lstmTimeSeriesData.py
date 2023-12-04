# LSTM on Airplane passengers Data from 1949-60

import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data"
filePath += "\\airline-passengers.csv"
df = pd.read_csv(filePath, usecols= [1], skipfooter=3)

# Visualize
plt.figure(figsize=(15, 5))
plt.plot(df, label = "Airline Passengers")
plt.xlabel("Months")
plt.ylabel("1000 International Airline Passengers")
plt.title("Monthly Total Airline Passengers 1949-1960")
plt.legend()
plt.show()


# Data preparation
#Get the raw data values from the pandas data frame
data_raw = df.values.astype("float32")

# We apply the MinMax scaler from sklearn
# to normalize data in the (0,1) interval
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(data_raw)

TRAIN_SIZE = 0.75
train_size = int(len(dataset)* TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Get data into shape to use in Keras
def create_dataset(dataset, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i+window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i+window_size, 0])
    return (np.array(data_X), np.array(data_Y))


# Create test and training sets for one-step-ahead regression
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)

# reshape the input data into appropriate form for Keras
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# LSTM Architecture
# One LSTM layer of 4 blocks
# One Dense layer to produce a single output
# Use MSE as loss function

from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(1, window_size)))
model.add(LSTM(units=4))
model.add(Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train_X, train_Y, epochs=100,
          batch_size=1, verbose=2)