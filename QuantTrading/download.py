import urllib.request

class NSEDataPreparation:
    """This class contains methods to download and prepare data from National Stock Exchange of India"""

    def __init__(self, startYear, endYear, dataFolderPath) -> None:
        self.startYear = startYear
        self.endYear = endYear
        self.localExtractedDataFilePath = dataFolderPath

    def constructNSEurl(self, securityType, day, month, year):
    # Ensure two-digit day
        if day < 10:
            day = "0"+ str(day)
        else:
            day = str(day)

        year = str(year)

        # securityType can be either "CM" or "FO"
        if securityType == "CM":
            nseURL = "https://archives.nseindia.com/content/historical/EQUITIES/"+year+"/"+month+"/"+"cm"+day+month+year+"bhav"+".csv.zip"
        elif securityType == "FO":
            nseURL = "https://archives.nseindia.com/content/historical/DERIVATIVES/"+year+"/"+month+"/"+"fo"+day+month+year+"bhav"+".csv.zip"
        else:
            nseURL = ""

        print("Construted URL:", nseURL)

        return nseURL


    def download(self, localDataFilePath, urlOfFile):
        """Method to download files in from url """

        hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
           'Accept': 'text/html, application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Cgarset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Language':'en-US,en;q=0.8',
           'Accept-Encoding': 'none',
           'Connection':'keep-alive'}
    
        webRequest = urllib.request.Request(urlOfFile)

        # The rest of our code will be enclosed within a try:/except: pair
        # This acts as a safety net in case we encounter some errors when assessing the web urls or working with files
        try:
            page = urllib.request.urlopen(webRequest)

            content = page.read()

            with open(localDataFilePath, "wb") as output:
                output.write(bytearray(content))

        except urllib.request.HTTPError:
            print(urllib.request.HTTPError.fp.read())

    

    def unzip(self, localDataFilePath):
        """Method to unzip the compressed zip file"""
        import os
        
        if os.path.exists(localDataFilePath):    #Checking if the file to be unzipped exists
            listOfFiles = []   #Zip file might contain multiple files, so we maintain a list of all extracted files

            with open(localDataFilePath, "rb") as fileHandler:
                import zipfile
                zipfileHandler = zipfile.ZipFile(fileHandler)
                # This zipHandler will be able to access and do stuff with files inside our zip file
                for name in zipfileHandler.namelist():   #Iterating through each file in the zip file
                    zipfileHandler.extract(name, self.localExtractedDataFilePath)
                    listOfFiles.append(self.localExtractedDataFilePath + name)

                print("Extracted", str(len(listOfFiles)), "files from", str(localDataFilePath))


    def downloadAllFiles(self):

        import time

        listOfMonths = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        listOfYears = [year for year in range(self.startYear, self.endYear+1)]

        for year in listOfYears:
            for month in listOfMonths:
                for dayOfMonth in range(31):
                    day = dayOfMonth + 1    # Adding 1 for date starting from 1
                    nseURL = self.constructNSEurl("CM", day, month, year)
                    strDay = str(day)
                    if day < 10:
                        strDay = "0" + str(day)

                    fileName = "cm" + strDay + month + str(year) + "bhav.csv.zip"
                    localFilePath = self.localExtractedDataFilePath + fileName
                    self.download(localFilePath, nseURL)
                    self.unzip(localFilePath)

                    time.sleep(10)




def download(filePath, urlOfFile):

    hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
           'Accept': 'text/html, application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Cgarset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Language':'en-US,en;q=0.8',
           'Accept-Encoding': 'none',
           'Connection':'keep-alive'}
    
    webRequest = urllib.request.Request(urlOfFile)

    # The rest of our code will be enclosed within a try:/except: pair
    # This acts as a safety net in case we encounter some errors when assessing the web urls or working with files

    try:
        page = urllib.request.urlopen(webRequest)

        content = page.read()

        with open(filePath, "wb") as output:
            output.write(bytearray(content))

    except urllib.request.HTTPError:
        print(urllib.request.HTTPError.fp.read())



# Cash markets
# https://archives.nseindia.com/content/historical/EQUITIES/2023/AUG/cm30AUG2023bhav.csv.zip

# Future markets
# https://archives.nseindia.com/content/historical/DERIVATIVES/2023/AUG/fo30AUG2023bhav.csv.zip

def constructNSEurl(secType, day, month, year):

# Ensure two-digit day
    if day < 10:
        day = "0"+ str(day)
    else:
        day = str(day)

    year = str(year)


    # secType can be either "CM" or "FO"
    if secType == "CM":
        nseURL = "https://archives.nseindia.com/content/historical/EQUITIES/"+year+"/"+month+"/"+"cm"+day+month+year+"bhav"+".csv.zip"
    elif secType == "FO":
        nseURL = "https://archives.nseindia.com/content/historical/DERIVATIVES/"+year+"/"+month+"/"+"fo"+day+month+year+"bhav"+".csv.zip"
    else:
        nseURL = ""

    return nseURL


def unzip(localFilePath, localExtractedFilePath):
    import os
    
    if os.path.exists(localFilePath):    #Checking if the file to be unzipped exists
        listOfFiles = []   #Zip file might contain multiple files, so we maintain a list of all extracted files

        with open(localFilePath, "rb") as fh:
            import zipfile
            zipfileHandler = zipfile.ZipFile(fh)
            # This zipHandler will be able to access and do stuff with files inside our zip file
            for name in zipfileHandler.namelist():   #Iterating through each file in the zip file
                zipfileHandler.extract(name, localExtractedFilePath)
                listOfFiles.append(localExtractedFilePath+name)

            print("Extracted", str(len(listOfFiles)), "files from", str(localFilePath))






