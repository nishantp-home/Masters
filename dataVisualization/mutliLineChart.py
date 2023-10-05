import matplotlib.pylab as plt
import pickle

with open('The Complete Python Data Visualization Course (Course Files)/Data Visualization - Source Code/data-viz/matplotlib/prog-langs-popularity.pickle', 'rb') as f:
    data = pickle.load(f)


#split into two lists
languages, rankings = zip(*data)

print(languages)
print(rankings)
#iterate over all the languages and call "plot" on their data
for i in range(len(languages)):
    year, ranking = zip(*rankings[i])
    plt.plot(year, ranking)


plt.xlabel('Year')
plt.ylabel('Ranking')
plt.title('Programming language ranking by year')
plt.legend(languages)
plt.show()