import nltk

# NLTK has many corpora and resources to explore natural language.
# A one-off run of nltk.download() will get you all the resources in one go.
# Once you've done that, you should have a repository of interesting texts including stuff like
# Moby Dick and an Inaugural  Address Corpus
from nltk.book import *

# These texts have now been loaded and you can refer them by their names.
# These are objects of type 'Text' and they have a bunch of cool methods

# Concordance will print all the occurrences of a word along with some context
text1.concordance('monstrous')
text2.concordance('monstrous')
# Authors of text1 and text2 use the word 'monstrous' in different connotations.
# Melville uses it for size and things that are terrifying, Austen uses it in a positive connotation

# Lets see what other words appear in the same context as 'monstrous'
text2.similar('monstrous')
# Clearly, Austen uses 'monstrous' to represent positive emotions and to amplify those emotions, 
# she seems to use it interchangeably with 'very'

text2.common_contexts(['monstrous', 'very'])


# Lets see how the usage of certain words by Presidents has changed over years
text4.dispersion_plot(['citizens', 'democracy','freedom', 'duties', 'America'])


# Often you want to extract features from a text
# These are attributes that will represent the text - words or sentences
# How do we split a piece of text into constituent sentences/words? 
# These are called tokens
from nltk.tokenize import word_tokenize, sent_tokenize

text="Mary had a little lamb. Her fleece was white as snow"
sents = sent_tokenize(text)
words = [word_tokenize(sent) for sent in sents]

# Lets filter out stopwords (words that are very common like 'was', 'a', 'as')
from nltk.corpus import stopwords
from string import punctuation

customStopWords = set(stopwords.words('english')+list(punctuation))

wordsWOStopwords = [word for word in word_tokenize(text) if word not in customStopWords]

text2 = 'Mary closed on closing night when she was in the mood to close.'
# 'close' appears in different morpholohical forms here,
# stemming will reduce all forms of the word 'close' to its root
# NLTK has multiple stemmers based on different rule/algorithms.
# Stemming is also known as lemmatization.
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]

# NLTK has functionality to automatically tag words as nouns, verbs, conjunctions etc.
nltk.pos_tag(word_tokenize(text2))