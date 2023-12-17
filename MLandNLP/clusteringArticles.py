# Download all the posts of a blog, and cluster them into 5 clusters
# Play around with the most common words in those clusters 
# to see how closely linked they are


from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from frequencySummarizer import FrequencySummarizer
from collections import defaultdict

# Function to download all posts on this blog
def getAllDoxyDonkeyPosts(url, links):
    page = urlopen(url).read()
    soup = BeautifulSoup(page)

    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                links.append(url)
                getAllDoxyDonkeyPosts(url, links)

        except:
            title = ""
    return

def getDoxyDonkyText(link, token):
    page = urlopen(link).read()
    soup = BeautifulSoup(page)

    text = soup.find('div', {'class': token}).text
    
    return text



blogUrl = "https://doxydonkey.blogspot.com/"
links = []
getAllDoxyDonkeyPosts(blogUrl, links=links)
doxyDonkeyPosts = {}
for link in links:
    doxyDonkeyPosts[link] = getDoxyDonkyText(link, 'post-body entry-content')


documentCorpus = []
# for onePost in doxyDonkeyPosts.values():
#     documentCorpus.append(onePost[0])

documentCorpus = list(doxyDonkeyPosts.values())


vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documentCorpus)
km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

keywords = defaultdict(int)
for i, cluster in enumerate(km.labels_):
    oneDocument = documentCorpus[i]
    fs = FrequencySummarizer()
    summary = fs.extractFeatures((oneDocument, ""),
                                 1000,
                                 [u"according", u"also", u"billion", u"like", u"new", u"one", u"year", u"first", u"last"])
    if cluster not in keywords:
        keywords[cluster] = set(summary)
    else:
        keywords[cluster] = keywords[cluster].intersection(set(summary))