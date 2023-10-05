#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle

# load data
with open('prog-langs-popularity.pickle', 'rb') as f:
    data = pickle.load(f)

# split into two lists
languages, rankings = zip(*data)

# iterate over all of the language and call "plot" on their data
for i in range(len(languages)):
    # for each language, split their data into years and rankings lists
    years, ranks = zip(*rankings[i])
    plt.plot(years, ranks)

# x-axis=year, y-axis=ranking, title=Rankings of Programming Languages
plt.xlabel('year')
plt.ylabel('ranking')
plt.title('Rankings of Programming Languages')
plt.legend(languages)
plt.show()
