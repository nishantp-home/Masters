from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section8"
filePath += "\\sonar.csv"
df = pd.read_csv(filePath, header=None)

dataset = df.values
# Split into input (X) and output (Y) variables
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# Encode y/class values as integars
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
 
# Baseline neural network model
def create_baseline():
    #Create model
    model = Sequential()
    model.add(Dense(60, kernel_initializer='normal', activation='relu')) 
    # 1 hidden layer with 60 neurons. Expects 60 inputs
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))  # output
    #Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_baseline, epochs=50, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

