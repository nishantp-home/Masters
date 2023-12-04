# Feed forward multilayer neural network

import pandas as pd
import numpy as np
import tensorflow as tf


# Load data from csv file
filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section9"
filePath += "\\Iris.csv"
iris = pd.read_csv(filePath)

# tensorFlow classifier requires data to be of type float32
# Cast data from float64 to float32
iris.iloc[:, 1:5] = iris.iloc[:, 1:5].astype(np.float32)
iris.dtypes

# Convert categorical variables to numerical
iris["Species"] = iris["Species"].map({"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, 1:5], iris["Species"], test_size=0.30, random_state=42)

# Build the DNN classifier
# Inputs: feature_columns (map the data to the model); hidden_units and n_classes
columns = iris.columns[1:5]
feature_columns = [tf.feature_column.numeric_column(k) for k in columns]

def input_fn(df, labels):
    feature_cols = {k: tf.constant(df[k].values, shape = [df[k].size, 1])  for k in columns}
    label = tf.constant(labels.values, shape= [labels.size, 1])
    return feature_cols, label

# Build 3 layer DNN with 10, 20, 10 units respectively
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

# train classifier
# classifier.fit(input_fn=lambda : input_fn(X_train, y_train), steps=1000)
classifier.train(input_fn=lambda : input_fn(X_train, y_train),steps=10000)

# Evaluate
ev = classifier.evaluate(input_fn=lambda: input_fn(X_test, y_test), steps=1)