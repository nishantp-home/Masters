import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score

# Load the data into a pandas dataframe
iris = sns.load_dataset('iris')
x = iris.sepal_length
y = iris.petal_length

lr = linear_model.LinearRegression()

degreeList = [1,2,3,4,5]

for deg in degreeList:
    lr.fit(np.vander(x, deg+1), y)
    y_1r = lr.predict(np.vander(x, deg+1))
    plt.plot(x, y_1r, label='degree'+str(deg))
    plt.legend(loc=2)
    print(r2_score(y,y_1r))

plt.plot(x, y, 'ok')