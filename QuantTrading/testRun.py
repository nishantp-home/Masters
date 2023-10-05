from download import download
from download import constructNSEurl, unzip
from download import NSEDataPreparation

# ticker = "^GSPC"
# startDate = "2015-01-01"
# endDate = "2016-01-01"
# freq = "d"

# nseFilePath = "E:/Eskills-Academy-projects/tutorials/Masters-DataSciencs/QuantTrading/dataSet/fm.zip"
# nseURL = constructNSEurl("FO",29,"AUG",2017)
# download(nseFilePath, nseURL)
# nseExtractedFilePath = "E:/Eskills-Academy-projects/tutorials/Masters-DataSciencs/QuantTrading/dataSet/"
# unzip(nseFilePath, nseExtractedFilePath)


dataFolderPath = "E:/Eskills-Academy-projects/tutorials/Masters-DataSciencs/QuantTrading/dataSet/"
sampleDataSet = NSEDataPreparation(2019, 2020, dataFolderPath)
sampleDataSet.downloadAllFiles()



