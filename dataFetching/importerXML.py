from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

url = 'https://apps.des.qld.gov.au/air-quality/xml/feed.php?category=1&region=ALL'
urlResult = urlopen(url)
rawData = urlResult.read()
xmlSoup = soup(rawData, features="xml")

southEastQueensland = xmlSoup.findAll('region', {'name': 'South East Queensland'})
southEastQueensland = southEastQueensland[0]

stations = southEastQueensland.findAll('station')
firstStation = stations[0]
sd = firstStation.findAll('measurement', {'name':'Nitrogen dioxide'})


headers = 'Station Name, Nitrogen dioxide, Ozone, Sulfur Dioxide, Carbon Monoxide, Particle PM10, Particle PM2.5, Particles TSP, Visibility'

for station in stations:
    stationString = ''
    stationName = station['name']
    nd = station.findAll('measurement', {'name': 'Nitrogen Dioxide'})
    if len(nd) == 0:
        nd = ''
    else:
        nd = nd[0].text