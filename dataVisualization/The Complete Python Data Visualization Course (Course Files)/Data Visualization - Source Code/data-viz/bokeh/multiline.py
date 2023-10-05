#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
import pickle

output_file('multiline.html')

with open('prog-langs-popularity.pickle', 'rb') as f:
    data = pickle.load(f)

languages, rankings = zip(*data)

fig = figure(x_axis_label='year', y_axis_label='rank', title='Rankings of Programming Languages')

for i in range(len(languages)):
    years, ranks = zip(*rankings[i])
    # legend & color for this particular line
    fig.line(years, ranks, line_width=2, legend=languages[i], color=palette[i])

# interactive legend:
fig.legend.click_policy = 'hide'

show(fig)
