import matplotlib.pylab as plt
import pickle

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/prog-langs-popularity.pickle', 'rb') as f:
    data = pickle.load(f)

#split into two lists
languages , rankings = zip(*data)
print(languages)
print(rankings)

javaYears, javaRanks = zip(*rankings[0])

plt.plot(javaYears, javaRanks)

plt.xticks(javaYears)
plt.title('Java rankings by year')
plt.xlabel('Year')
plt.ylabel('Ranking')
plt.show()