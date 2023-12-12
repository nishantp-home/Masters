import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

# load stopwords
# selected after observing results without stopwords
stopwords = [
  'the',
  'about',
  'an',
  'and',
  'are',
  'at',
  'be',
  'can',
  'for',
  'from',
  'if',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'this',
  'to',
  'you',
  'your',
  'with',
]

# find urls and twitter usernames within a string
url_finder = re.compile(r"(?:\@|https?\://)\S+")


def filter_tweet(s):
    s = s.lower()   # downcase each tweet
    s = url_finder.sub("", s)  # remove urls and usernames with blanks
    return s


# Load data
filePath = "E:\\Eskills-Academy-projects\\lazyProgrammerCoursesData\\machine_learning_examples-unsupervised_class\\"
fileName = "tweets.csv"
file = filePath + fileName
df = pd.read_csv(file)
text = df.text.tolist()
text = [filter_tweet(s) for s in text]

#transform the text into a data matrix
tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()


# Subsample for efficiency
# remember: calculating distances is O(N**2)
N = X.shape[0]
idx = np.random.choice(N, size=2000, replace=False)
x = X[idx]
labels = df.handle[idx].tolist()

# proportions of each label
# so we can be confident that each is represented equally
pTrump = sum(1.0 if e == 'realDonaldTrump' else 0.0 for e in labels) / len(labels)
print("Proportion of @realDonaldTrump tweets: %.3f" % pTrump)


# Transform the data matrix into a pairwise distances list
dist_array = pdist(x)

# Calculate hierarchy
Z = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(Z, labels=labels)
plt.show()

# Other linkages:
    # 'Single' would fail
    # 'Complete' wont produce good results


# Convert labels to (1,2), not (0,1)
# Since that's what's returned by fcluster
Y = np.array([1 if e == 'realDonaldTrump' else 2 for e in labels])

# Get cluster assignments
# Threshold 9 was chosen empirically to yoield 2 clusters
C = fcluster(Z, 9, criterion='distance')  # returns 1,2,...K
categories = set(C)
print("values in C:", categories)


def purity(true_labels, cluster_assignments, categories):
    # maximum purity is 1, higher the better
    N = len(true_labels)

    total = 0.0
    for k in categories:
        max_intersection = 0
        for j in categories:
            intersection = ((cluster_assignments==k) & (true_labels==j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection

    total /= N

    return total

print("Purity:", purity(Y, C, categories))

# One of the cluster is by Donald trump
# And this cluster is very small

if (C==1).sum() < (C==2).sum():
    d, h = 1, 2
else:
    d, h = 2, 1

actually_donald = ((C==d) & (Y==1)).sum()
donald_cluster_size = (C==d).sum()
print("Purity of @realDonaldTrump cluster:", float(actually_donald) / donald_cluster_size)

actually_hillary = ((C==h) & (Y==2)).sum()
hillary_cluster_size = (C==h).sum()
print("Purity of @HillaryClinton cluster:", float(actually_hillary) / hillary_cluster_size)


# How would a classifier do ?
# note: classification is always easier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, df.handle)
print("Classifier score:", rf.score(X, df.handle))


# what words have the highest tf-idf in cluster 1? in cluster 2?
w2i = tfidf.vocabulary_

# tf-idf vectorizer todense() returns a matrix rather than array
# matrix always wants to be 2-D, so we convert to array in order to flatten
d_avg = np.array(x[C == d].mean(axis=0)).flatten()
d_sorted = sorted(w2i.keys(), key=lambda w: -d_avg[w2i[w]])

print("\nTop 10 'Donald cluster' words:")
print("\n".join(d_sorted[:10]))

h_avg = np.array(x[C == h].mean(axis=0)).flatten()
h_sorted = sorted(w2i.keys(), key=lambda w: -h_avg[w2i[w]])

print("\nTop 10 'Hillary cluster' words:")
print("\n".join(h_sorted[:10]))
