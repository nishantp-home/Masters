from bokeh.io import show
from bokeh.plotting import figure
import pickle

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/fruit-sales.pickle', 'rb') as f:
    data = pickle.load(f)

fruit, soldCount = zip(*data)

plot = figure(x_range= fruit, y_axis_label='Fruit sold (millions)', title = 'Fruit sold by year')
plot.vbar(x=fruit, top=soldCount, width=0.9)

show(plot)