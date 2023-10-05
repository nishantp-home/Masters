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

# scatter, reg, kde, hex
axes = sns.jointplot(sepal_length, sepal_width, kind='hex')
axes.set_axis_labels('Sepal length (cm)', 'Sepal width (cm)')
plt.show()
