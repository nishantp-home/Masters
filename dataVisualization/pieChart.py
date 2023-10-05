import matplotlib.pylab as plt
import pickle

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/devs-outside-time.pickle', 'rb') as f:
    data = pickle.load(f)

# 
time, responses = zip(*data)

plt.pie(responses, labels = time, autopct='%.1f%%')
plt.title('Daily Time Developers spend outside')
plt.axis('equal')

plt.show()