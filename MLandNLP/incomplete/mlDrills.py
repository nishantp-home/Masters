import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from urllib.request import urlopen, Request

from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from selenium import webdriver
from frequencySummarizer import FrequencySummarizer


def getWashPostText(url, token):
    """Takes the URL of an article in the Washington Post,
       and then returns the article minus all of the crud - HTML, javascript, etc. How?
       By searching everything that lies between the tags titled 'token' 
       Like most web-scraping, this only works for urls where we know the structure.
       This will also change from time-to-time as different HTML formats are employed in the website"""

    try:
        page = urlopen(url).read().decode('utf8')
    except:
        # If unable to download the url, return title = None, article = None
        return (None, None)

    soup = BeautifulSoup(page)
    if soup is None:
        return (None, None)

    text = ""
    if soup.find_all(token) is not None:
        # Search the page for whatever token demarcates the article
        # usually '<article></article>'
        text = ''.join(map(lambda p: p.text, soup.find_all(token)))
        # Mush together all the text in the '<article></article>' tags
        soup2 = BeautifulSoup(text)
        # create a soup object of the text within the <article> tags
        if soup2.find_all('p') is not None:
            # Now mush together the contents of what is in <p> </p>
            text = ''.join(map(lambda p: p.text, soup2.find_all('p')))

    return text, soup.title.text

# def getNYTText(url, token=None):
#     response = requests.get(url)
#     # This is an alterantive way to get the contents of a URL
#     soup = BeautifulSoup(response.content)
#     page = str(soup)
#     title = soup.find('title').text
#     mydivs = soup.findAll("p", {"class":"story-body-text story-content"})
#     text = ''.join(map(lambda p: p.text, mydivs))

#     return text, title


# Function that will take the URL of the entire section of a newspaper - say the Technology or Sports section
# and parse all the URLs for the articles linked off that section.
# These sections also come with plenty of non-news links- 'about',
# how to syndicate etc., so we will employ a little hack - we will consider
# something to be news article only if the url has a datelin. This is actually safe-
# Its pretty much the rule for articles to have a date, and virtually all important newspapers mush this date into the URL.

def scrapeSource(url, magicFrag='2015', scraperFunction=getWashPostText, token='None'):
    urlBodies = {}

    page = urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(page)
    # Find the links
    # Remember that links are always of the form
    # <a href='link-url'> link-text </a>
    numErrors = 0
    for a in soup.findAll('a'):
        try:
            url = a['href']
            if ((url not in urlBodies) and
               ((magicFrag is not None and magicFrag in url)
                    or magicFrag is None)):
                body = scraperFunction(url, token)
                if body and len(body) > 0:
                    urlBodies[url] = body

        except:
            numErrors += 1





def getDoxyDonkey(testUrl, token):
    response = requests.get(testUrl)
    soup = BeautifulSoup(response.content)
    page = str(soup)
    title = soup.find('title').text
    mydivs = soup.findAll('div', {'class': token})
    text = ''.join(map(lambda p: p.text, mydivs))
    return text, title


testUrl = 'https://doxydonkey.blogspot.com/'
testArticle = getDoxyDonkey(testUrl=testUrl, token="post-body")
fs = FrequencySummarizer()
testArticleSummary = fs.extractFeatures(testArticle, 25)




urlWPnonTech = "https://www.washingtonpost.com/sports/"
urlWPTech = "https://www.washingtonpost.com/business/technology/"

wpTechArticles = scrapeSource(urlWPTech, '2015', getWashPostText, 'article')
wpNonTechArticles = scrapeSource(
    urlWPnonTech, '2015', getWashPostText, 'article')

# Collect article summaries in an easy to classify form
articleSummaries = {}

for articleUrl in wpTechArticles:
    if len(wpTechArticles[articleUrl][0]) > 0:
        fs = FrequencySummarizer()
        summary = fs.extractFeatures(wpTechArticles[articleUrl], 25)
        articleSummaries[articleUrl] = {'featrue-vector': summary,
                                        'label': 'Tech'}

for articleUrl in wpNonTechArticles:
    if len(wpNonTechArticles[articleUrl][0]) > 0:
        fs = FrequencySummarizer()
        summary = fs.extractFeatures(wpNonTechArticles[articleUrl], 25)
        articleSummaries[articleUrl] = {'featrue-vector': summary,
                                        'label': 'Non-Tech'}

# K-nearest neighbour algorithm
# Find the 5 nearest (most similar) articles, and then
# carry out a majority vote of those 5.
similarities = {}
for articleUrl in articleSummaries:
    oneArticleSummary = articleSummaries[articleUrl]['feature-vector']
    similarities[articleUrl] = len(set(testArticleSummary).intersection(set(oneArticleSummary)))

labels = defaultdict(int)
knn = nlargest(5, similarities, key=similarities.get)
for oneNeighbor in knn:
    labels[articleSummaries[oneNeighbor]['label']] += 1

nlargest(1, labels, key=labels.get)



# Classification using Naive-Bayes classification
