#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# load data
with open('coding-exp-by-dev-type.pickle', 'rb') as f:
    data = pickle.load(f)

# split into two lists
dev_types, years_exp = zip(*data)
dev_types = list(dev_types)
years_exp = list(years_exp)

# matplotlib: plt.barh(bar_coords, years_exp)
sns.barplot(y=dev_types, x=years_exp)
plt.xlabel('years')
plt.title('Years of Coding Experience by Developer Type')
plt.tight_layout()
plt.show()
