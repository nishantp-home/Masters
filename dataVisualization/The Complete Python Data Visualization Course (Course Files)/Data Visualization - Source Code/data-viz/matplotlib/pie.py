#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle

# load data
with open('devs-outside-time.pickle', 'rb') as f:
    data = pickle.load(f)

# split into two lists
time, responses = zip(*data)

plt.pie(responses, labels=time, autopct='%d%%')
# force the x/y axes to have the same scale
# circle instead of an ellipse
plt.axis('equal')
plt.title('Daily Time Developers Spend Outside')
plt.show()
