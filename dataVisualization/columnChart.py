import matplotlib.pyplot as plt
import pickle

# load our data
with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/fruit-sales.pickle', 'rb') as f:
    data = pickle.load(f)

#splitting a list of tuples into two lists
fruit, soldCount = zip(*data)
barCoords = range(len(fruit))
plt.bar(barCoords, soldCount)
plt.xticks(barCoords, fruit)
plt.ylabel('Number of fruits  (millions)')
plt.title('Number of fruits sold (2017)')
plt.show()

