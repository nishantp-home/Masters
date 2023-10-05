#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
import pickle

output_file('scatter.html')

# load data
with open('iris.pickle', 'rb') as f:
    iris = pickle.load(f)

# load sepal length and sepal width for all classes
sepal_length = iris['data'][:, 0]
sepal_width = iris['data'][:, 1]
classes = iris['target']

# separate data via class
setosa_sepal_length = sepal_length[classes == 0]
setosa_sepal_width = sepal_width[classes == 0]
versicolor_sepal_length = sepal_length[classes == 1]
versicolor_sepal_width = sepal_width[classes == 1]
virginica_sepal_length = sepal_length[classes == 2]
virginica_sepal_width = sepal_width[classes == 2]

fig = figure(x_axis_label='Sepal length (cm)', y_axis_label='Sepal width (cm)')
fig.circle(setosa_sepal_length, setosa_sepal_width, color=palette[0], legend='setosa')
# plot versicolor sepal length v. width
fig.circle(versicolor_sepal_length, versicolor_sepal_width, color=palette[1], legend='versicolor')
# plot virginica sepal length v. width
fig.circle(virginica_sepal_length, virginica_sepal_width, color=palette[2], legend='virginica')

show(fig)
