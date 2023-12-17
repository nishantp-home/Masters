# Objective : Take in the url of a newspaper article (from the washington post)
# And automatically summarie it in 3 sentences

# Step 1: Download the contents of the url  -> regular expressions
# Step 2: Extract the article from all the other html that is in the webpage   -> BeautifulSoup
# Step 3: Figure out which are the 3 most important sentences in the article    -> nltk
    # 1. Find the most common words in the article  (tokenize, eliminate stopwords, find frequency of each word, find word score, rank words)
    # 2. Find the sentence in which those most common words occur most often
    # 3. Thats the most important sentence

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from collections import defaultdict
from bs4 import BeautifulSoup
from urllib.request import urlopen
from heapq import nlargest
import re
import numpy as np


def getText(url, showArticle=False):
    """Input: url, returns the article without all of the crud (i.e. html, javascript), and its title"""
    try:
        article = urlopen(url=url).read().decode('utf8')
    except:
        return (None, None)
    
    parsedArticle = BeautifulSoup(article, 'lxml')  
    if parsedArticle == None:
        return (None, None)
    
    articleTitle = parsedArticle.title.text

    paragraphs = parsedArticle.find_all('p')
    articleText = ""
    for paragraph in paragraphs:
        articleText += paragraph.text

    # Text Preprocessing: 
    # Remove square brackets, extra spaces, digits        
    articleText = re.sub(r'\[|[0-9]|\]', '', articleText)   # square brackets, digits
    articleText = re.sub(r'\s+', ' ', articleText)      # extra spaces
    
    if showArticle is True:
        print(articleTitle + "\n")
        print(articleText)

    return articleTitle, articleText

def key_to_index(mydict, list_of_keys):

    dictIndices = defaultdict(int)
    for key1 in list_of_keys:
        for index, key2 in enumerate(sorted(mydict.keys())):
            if key1 == key2:
                dictIndices[key1] = index

    return list(dictIndices.values())
    

        
class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9) -> None:
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english')+
                              list(punctuation)+
                              [u"'s",'"'])
        

    def _compute_frequencies(self, word_sent, customStopWords=None):
        """Method takes in a list of sentences, and returns a dictionary 
        of where keys are words, and values are the frequencies of those
        words in the set of sentences"""

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

        #Normalize frequency b/w 0-1
        maxFreq = float(max(freq.values()))
        fequencyKeys = freq2.keys()
        for word in fequencyKeys:
            freq[word] = freq[word] / maxFreq
            # Filter out frequencies that are too high (> max_cut) or too low (< min_cut)
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                del freq[word]

        return freq
    
        
    def summarize(self, article, n, showSummary= False):

        title = article[0]
        text = article[1]

        sentences = sent_tokenize(text=text)  # list of sentences
        assert n <= len(sentences)
        word_sent = [word_tokenize(s.lower()) for s in sentences]  # list of sentences 
                                                                    # with each sentance as a list of words

        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)  # dictionary for rankings of sentences

        for i, sentence in enumerate(word_sent):
            for word in sentence:
                if word in self._freq:
                    ranking[i] += self._freq[word]

        topN_rankingKeys = nlargest(n, ranking, key=ranking.get)  # gets the list of n-keys with largest values in descending order ()using ranking.get method 
        sentence_index = key_to_index(ranking, topN_rankingKeys)
        
        summary = ""
        for i in sentence_index:
            summary += sentences[i] 

        if showSummary is True:
            print("Summary:")
            print(summary)
        
        return summary
        

url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
# url = "https://www.washingtonpost.com/news/the-switch/wp/2015/10/15/amid-the-adblockalypse-advertisers-apologize-for-messing-up-the-web/"
nlpArticle = getText(url=url, showArticle=True)

summ = FrequencySummarizer()
summary = summ.summarize(nlpArticle, 3, showSummary=True)
