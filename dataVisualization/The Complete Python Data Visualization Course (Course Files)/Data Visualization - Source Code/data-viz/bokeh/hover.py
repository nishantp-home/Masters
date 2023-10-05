#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bokeh.io import show, output_file
from bokeh.plotting import figure
import pickle

output_file('hover.html')

with open('coding-exp-by-dev-type.pickle', 'rb') as f:
    data = pickle.load(f)
    
dev_types, years_exp = zip(*data)
data_source = {'dev_types': dev_types, 'years_exp': years_exp}

# tool tips: "years of experience: actual val"
TOOLTIPS = [('years of experience', '@years_exp')]
plot = figure(y_range=dev_types, x_axis_label='years', title='Coding Experience by Developer Type', tools='hover', tooltips=TOOLTIPS)
# set data source: looks up values in data source dictionary
plot.hbar(y='dev_types', right='years_exp', height=0.9, source=data_source)

show(plot)
