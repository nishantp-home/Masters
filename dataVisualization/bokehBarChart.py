from bokeh.io import show, output_file
from bokeh.plotting import figure
import pickle

output_file('hover.html')

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/coding-exp-by-dev-type.pickle', 'rb') as f:
    data = pickle.load(f)

devTypes, yearsExpCount = zip(*data)
dataSource = {'devtypes': devTypes, 'yearsExpCount': yearsExpCount}

TOOLTIPS = [('years of experience', '@yearsExpCount')]

plot = figure(y_range=devTypes, x_axis_label='Years', title='Coding experience by years')

plot.hbar(y=devTypes, right=yearsExpCount, height=0.9)
show(plot)