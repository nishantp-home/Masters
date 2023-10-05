#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# load data
with open('iris.pickle', 'rb') as f:
    iris = pickle.load(f)

# extract the first column from the data table (get all of the rows)
sepal_length = iris['data'][:, 0]
sepal_width = iris['data'][:, 1]
classes = iris['target']

# matplotlib: plt.scatter(sepal_length, sepal_width, c=classes)
sns.scatterplot(sepal_length, sepal_width, hue=classes, legend=False)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris data: sepal length v. width')
plt.show()
