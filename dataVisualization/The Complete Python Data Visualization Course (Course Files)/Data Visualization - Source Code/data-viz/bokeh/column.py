#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bokeh.io import show
from bokeh.plotting import figure
import pickle

# load data
with open('fruit-sales.pickle', 'rb') as f:
    data = pickle.load(f)

# split into two lists
fruit, num_sold = zip(*data)

plot = figure(x_range=fruit, y_axis_label='Fruit sold (millions)', title='Fruit sold (2017)')
plot.vbar(x=fruit, top=num_sold, width=0.9)

show(plot)
