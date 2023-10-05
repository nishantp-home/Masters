import matplotlib.pyplot as plt
import pickle

# load data 
with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/coding-exp-by-dev-type.pickle', 'rb') as f:
    data = pickle.load(f)

print(data)

#split into two lists
devTypes , yearsExpCount = zip(*data)

barCoords = range(len(devTypes))

plt.barh(barCoords, yearsExpCount)
plt.yticks(barCoords, devTypes, fontsize= 8)
plt.title('Years of experience by Developer type')

plt.tight_layout()
plt.show()