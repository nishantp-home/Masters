import numpy as np

# file 'site_data.csv' contains two-columns (start_page_id, end_page_id) of data
# ids can be (0...9) representing different pages of the site , start_page_id: -1 represents initial state
# end_page_id  B: Bound back, C: close page


# collect data

# empty dictionaries to fill the data
transitions = {}
rowSums = {}

# fill the dictionaries with count-data from the data file 'site_data.csv
for line in open('hiddenMarkovModel/site_data.csv'):
    s, e = line.rstrip().split(',')   #(s: start, e: end) is the key
    transitions[(s, e)] = transitions.get((s, e), 0.) + 1
    rowSums[s] = rowSums.get(s, 0.) + 1

# normalize
for k, v in transitions.items():
    s, e = k
    transitions[k] = v / rowSums[s]

# initial state distribution
print('Initial state distribution')
for k, v in transitions.items():
    s, e = k
    if s == '-1':
        print(e, ':', v)

# which page has the highest bounce ?
for k, v in transitions.items():
    s, e = k
    if e == 'B':
        print('Bounce rate for %s: %s' % (s, v))
