#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bokeh.io import show, output_file
from bokeh.plotting import figure
import pickle

output_file('bar.html')

with open('coding-exp-by-dev-type.pickle', 'rb') as f:
    data = pickle.load(f)
    
dev_types, years_exp = zip(*data)

plot = figure(y_range=dev_types, x_axis_label='years', title='Coding Experience by Developer Type')
plot.hbar(y=dev_types, right=years_exp, height=0.9)

show(plot)
