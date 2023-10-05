from bokeh.io import show, output_file
from bokeh.plotting import Figure
from bokeh.palettes import Dark2_5 as palette
from bokeh.layouts import Row, Column, gridplot
import pickle

output_file('multiplot.html')

#load data
with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/iris.pickle', 'rb') as f:
    iris = pickle.load(f)

# Load sepal length and width for all classes

sepalLength = iris['data'][:, 0]
sepalWidth = iris['data'][:, 1]
petalLength = iris['data'][:, 2]
petalWidth = iris['data'][:, 3]
classes = iris['target']

#Separate data via class
setosaSepalLength = sepalLength[classes == 0]
setosaSepalWidth = sepalWidth[classes == 0]
setosaPetalLength = petalLength[classes == 0]
setosaPetalWidth = petalWidth[classes == 0]

versicolorSepalLength = sepalLength[classes == 1]
versicolorSepalWidth = sepalWidth[classes == 1]
versicolorPetalLength = petalLength[classes == 1]
versicolorPetalWidth = petalWidth[classes == 1]

virginicaSepalLength = sepalLength[classes == 2]
virginicaSepalWidth = sepalWidth[classes == 2]
virginicaPetalLength = petalLength[classes == 2]
virginicaPetalWidth = petalWidth[classes == 2]


fig1 = Figure(x_axis_label='Sepal length [cm]', y_axis_label='Sepal width [cm]')
fig1.circle(setosaSepalLength, setosaSepalWidth, color= palette[0], legend_label='setosa')
fig1.circle(versicolorSepalLength, versicolorSepalWidth, color= palette[1], legend_label='versicolor')
fig1.circle(virginicaSepalLength, virginicaSepalWidth, color= palette[2], legend_label='virginica')

fig2 = Figure(x_axis_label='Petal length [cm]', y_axis_label='Petal width [cm]')
fig2.circle(setosaPetalLength, setosaPetalWidth, color= palette[0], legend_label='setosa')
fig2.circle(versicolorPetalLength, versicolorPetalWidth, color= palette[1], legend_label='versicolor')
fig2.circle(virginicaPetalLength, virginicaPetalWidth, color= palette[2], legend_label='virginica')

fig3 = Figure(x_axis_label='Sepal length [cm]', y_axis_label='Petal Length [cm]', x_range= fig1.x_range)
fig3.circle(setosaSepalLength, setosaPetalLength, color= palette[0], legend_label='setosa')
fig3.circle(versicolorSepalLength, versicolorPetalLength, color= palette[1], legend_label='versicolor')
fig3.circle(virginicaSepalLength, virginicaPetalLength, color= palette[2], legend_label='virginica')

fig4 = Figure(x_axis_label='Sepal Width [cm]', y_axis_label='Petal Width [cm]')
fig4.circle(setosaSepalWidth, setosaPetalWidth, color= palette[0], legend_label='setosa')
fig4.circle(versicolorSepalWidth, versicolorPetalWidth, color= palette[1], legend_label='versicolor')
fig4.circle(virginicaSepalWidth, virginicaPetalWidth, color= palette[2], legend_label='virginica')

show(gridplot([[fig1, fig2],[fig3, fig4]]))