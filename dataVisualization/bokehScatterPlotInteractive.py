from bokeh.io import show, output_file
from bokeh.plotting import Figure
from bokeh.palettes import Dark2_5 as palette
import pickle

output_file('scatter.html')

#load data
with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/iris.pickle', 'rb') as f:
    iris = pickle.load(f)

# Load sepal length and width for all classes

sepalLength = iris['data'][:, 0]
sepalWidth= iris['data'][:, 1]
classes = iris['target']

#Separate data via class
setosaSepalLength = sepalLength[classes == 0]
setosaSepalWidth = sepalWidth[classes == 0]
versicolorSepalLength = sepalLength[classes == 1]
versicolorSepalWidth = sepalWidth[classes == 1]
virginicaSepalLength = sepalLength[classes == 2]
virginicaSepalWidth = sepalWidth[classes == 2]

fig = Figure(x_axis_label='Sepal length [cm]', y_axis_label='Sepal width [cm]')
fig.circle(setosaSepalLength, setosaSepalWidth, color= palette[0], legend='setosa')
fig.circle(versicolorSepalLength, versicolorSepalWidth, color= palette[1], legend='versicolor')
fig.circle(virginicaSepalLength, virginicaSepalWidth, color= palette[2], legend='virginica')

show(fig)
