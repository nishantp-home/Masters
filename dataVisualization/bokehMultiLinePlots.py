from bokeh.io import show, output_file
from bokeh.plotting import Figure
from bokeh.palettes import Dark2_5 as palette
import pickle

output_file('multiline.html')

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/prog-langs-popularity.pickle', 'rb') as f:
    data = pickle.load(f)

languages, rankings = zip(*data)
fig = Figure(x_axis_label='year', y_axis_label='ranking', title='Rank vs Year')

for i in range(len(languages)):
    years, ranks = zip(*rankings[i])
    #legend and color for this particular line
    fig.line(years, ranks, line_width=2, legend=languages[i], color=palette[i])

# Interactive legend:
fig.legend.click_policy = 'hide'
show(fig)