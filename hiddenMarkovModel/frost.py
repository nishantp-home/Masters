# This application builds a second order language model and generate phrases

import numpy as np
import string

# Three dictionaries
initial = {}   #stores the probability distribution for start of a phrase/sentance
secondWord = {}   # stores the distribution for second word of a phrase/sentance
transitions = {}  # stores all the second order transitions

def removePunctuation(s):
    """Takes in a string s, returns the string with removed punctations"""
    return s.translate(string.punctuation)

def add2dict(d, k, v):
    """Takes in a dictionary: d, key:k, and a value: v, appends the k, v pair to the dictionary if it doesnot exist"""
    if k not in d:
        d[k] = []    #creates an empty array
    d[k].append(v)


fileNameWithPath = 'hiddenMarkovModel/robert_frost.txt'

for line in open(fileNameWithPath):
    tokens = removePunctuation(line.rstrip().lower()).split()

    T = len(tokens)  # length of the sequence
    for i in range(T):   #loop through every token in the sequence
        t = tokens[i]
        if i == 0:    # that means we are looking at the first word
            initial[t] = initial.get(t, 0.) + 1     # we are keeping counts of each word appearing as the first word of the sentance
        else:     # otherwise we need to get the previous word
            t_1 = tokens[i-1]       # t-1
            if i == T - 1:   # we are looking at the last word
                add2dict(transitions, (t_1, t), 'END')
            if i == 1:  # we are looking at the second word of the sentance
                add2dict(secondWord, t_1, t)    # add second word to dictionary
            else: 
                t_2 = tokens[i-2]
                add2dict(transitions, (t_2, t_1), t)


# Normalize the distribution
initialTotal = sum(initial.values()) 
for t, c in initial.items():
    initial[t] = c / initialTotal


def list2pdict(ts):
    """Takes in a list ts, turns it into and returns a dictionary of probabilities"""
    d = {}  # initialize an empty dictionary
    n = len(ts)   # total number of values
    for t in ts:
        d[t] = d.get(t, 0.) + 1   # frequency count of all values

    # We go through the dictionary
    for t, c in d.items():    
        d[t] = c / n     # Divide frequency by the total number of values

    return d

for t_1, ts in secondWord.items():
    secondWord[t_1] = list2pdict(ts)

for k, ts in transitions.items():
    transitions[k] = list2pdict(ts)


# Sampling a word from this dictionary of proababilities
def sampleWord(d):
    """Takes in a dictionary: d"""
    p0 = np.random.random()   #generates a random number between 0 and 1
    cumulative = 0     # keeps a cumulative count of all probabilities seen so far
    for t, p in d.items():
        cumulative +=p
        if p0 < cumulative:
            return t
    assert(False)    # To make sure that the code should never goes here
        

def generate(sentanceCount):

    for i in range(sentanceCount):
        sentance = []

        w0 = sampleWord(initial)   #sample the first word
        sentance.append(w0)

        w1 = sampleWord(secondWord[w0])
        sentance.append(w1)

        while True:   #Infinite loop
            w2 = sampleWord(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sentance.append(w2)
            w0 = w1
            w1 = w2

        print(' '.join(sentance))



generate(5)

