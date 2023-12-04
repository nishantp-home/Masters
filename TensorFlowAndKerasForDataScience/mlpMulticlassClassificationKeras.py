import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# load dataset
filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section8"
filePath += "\\iris1.csv"
df = pd.read_csv(filePath, header=None)

dataset = df.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:,4]


#encode the response variable
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# Convert integars to dummy variables (i.e. one hot encoded)
dummy_Y = np_utils.to_categorical(encoded_Y)

# Define neural network 
# 4 inputs -> [8 hidden nodes] -> 3 outputs

# definie the baseline model
def baseline_model():
    # Create model
    model = Sequential()
    model.add(InputLayer(input_shape=(4,)))
    model.add(Dense(8, activation='relu'))    # Check if we need to explicitly provide input layer dimensions
    model.add(Dense(3, activation='softmax'))
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, verbose=0)

# Use kfold of scikit learn
kfold = KFold(n_splits=10, shuffle=True, random_state=5)
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))