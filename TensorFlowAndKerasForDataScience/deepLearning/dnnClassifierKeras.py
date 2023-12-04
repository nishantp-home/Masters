from keras.models import Sequential
from keras.layers import Dense, InputLayer
import numpy as np

filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section9"
filePath += "\\pima-indians-diabetes.csv"
dataset = np.loadtxt(filePath, delimiter=",")

# Split data into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Add multiple hidden layers
model = Sequential()
model.add(InputLayer(input_shape=(8,)))
# First hidden layer with 12 neurons and expects 8 input variables
model.add(Dense(12, activation="relu"))
# Second hidden layer has 8 neurons
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Output layer has 1 neuron to predict the class

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))