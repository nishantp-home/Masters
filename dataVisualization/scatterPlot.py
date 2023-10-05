import matplotlib.pylab as plt
import pickle

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/iris.pickle', 'rb') as f:
    iris = pickle.load(f)


sepalLength = iris['data'][:,0]
sepalWidth = iris['data'][:,1]
petalLength = iris['data'][:,2]
petalWidth = iris['data'][:,3]
classes = iris['target']

fig, axes = plt.subplots(2, 2)
axes[0,0].scatter(sepalLength, sepalWidth, c=classes)
axes[0,0].set_xlabel('Sepal Length [cm]')
axes[0,0].set_ylabel('Sepal Width [cm]')

axes[0,1].scatter(petalLength, petalWidth, c=classes)
axes[0,1].set_xlabel('Petal Length [cm]')
axes[0,1].set_ylabel('Petal Width [cm]')

axes[1,0].scatter(sepalLength, petalLength, c=classes)
axes[1,0].set_xlabel('Sepal Length [cm]')
axes[1,0].set_ylabel('Petal Length [cm]')

axes[1,1].scatter(sepalWidth, petalWidth, c=classes)
axes[1,1].set_xlabel('Sepal Width [cm]')
axes[1,1].set_ylabel('Petal Width [cm]')


fig.suptitle('Iris dataset')
plt.tight_layout()

plt.show()

# plt.scatter(sepalLength, sepalWidth, c=classes)
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Sepal width [cm]')
# plt.title('Sepal length vs width')
# plt.show()