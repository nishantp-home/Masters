import pandas as pd

# Importing tablular data from urls 
urlAmericaStatesList = 'https://simple.wikipedia.org/wiki/List_of_U.S._states'
americaStatesList = pd.read_html(urlAmericaStatesList)
americaStatesList = americaStatesList[0]

urlFailedBankList = "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/"
failedBankList = pd.read_html(urlFailedBankList)
failedBankList=failedBankList[0]

urlWHSitesPhillipines = "https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_the_Philippines"
whsPhillipines = pd.read_html(urlWHSitesPhillipines)

# Importing data from JSON files
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section2\\skorea.json'
df = pd.read_json(filePath)