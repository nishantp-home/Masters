from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from string import punctuation
from heapq import nlargest

class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9) -> None:
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') +
                              list(punctuation) +
                              [u"'s", '"'])

        # Notice that stopwards are set, not a list.
        # its easy to go from set to a list, and vice-versa.
        # (simply use set() and list() functinos)
        # But conceptually, sets are different from lists.
        # Sets don't have ordering to their elements, while lists do

    def _compute_frequencies(self, word_sent, customStopWords=None):
        freq = defaultdict(int)
        if customStopWords is None:
            stopwords = set(self._stopwords)
        else:
            stopwords = set(customStopWords).union(self._stopwords)

        for sentence in word_sent:
            for word in sentence:
                if word not in stopwords:
                    freq[word] += 1

        freq2 = freq.copy()
        fequencyKeys = freq2.keys()

        m = float(max(freq.values()))
        for word in fequencyKeys:
            freq[word] = freq[word] / m
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                del freq[word]

        return freq

    def extractFeatures(self, article, n, customStopWords=None):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent, customStopWords)

        if n < 0:
            return nlargest(len(self._freq.keys()), self._freq, key=self._freq.get)
        else:
            return nlargest(n, self._freq, key=self._freq.get)

    def extractRawFrequencies(self, article):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        freq = defaultdict(int)
        for sentence in word_sent:
            for word in sentence:
                if word not in self._stopwords:
                    freq[word] += 1
        return freq

    def summarize(self, article, n):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent=word_sent)
        ranking = defaultdict(int)
        for i, s in enumerate(word_sent):
            for w in s:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sentences_index = nlargest(n, ranking, key=ranking.get)

        return [sentences[j] for j in sentences_index]