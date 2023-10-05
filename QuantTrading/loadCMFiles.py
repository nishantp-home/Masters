import mysql.connector
from pathlib import Path

config = {
    'user': 'root',
    'password': 'abc123',
    'host':'127.0.0.1',
    'database':'nse'
}

conn = mysql.connector.connect(**config)
c = conn.cursor()

c.execute("use nse")
#c.execute("create TABLE cmStaging(symbol VARCHAR(256),series VARCHAR(256),open FLOAT,high FLOAT,low FLOAT,close FLOAT,last FLOAT,prevclose FLOAT,tottrdqty FLOAT,tottrdval FLOAT,timestamp date,totaltrades FLOAT,isin VARCHAR(256))")

def insertRows(fileName, c):
    delimiter =r','

    dateString=r'%d-%b-%Y'

    c.execute("Load data infile %s into table cmStaging fields terminated by %s ignore 1 lines"
              "(symbol, series, open, high, low, close, last, prevclose, tottrdqty,tottrdval, @timestamp, totaltrades, isin)"
              "SET timestamp = STR_TO_DATE(@timestamp, %s)", (fileName, delimiter, dateString))
              



localExtractFilePath = r'E:/Eskills-Academy-projects/tutorials/Masters-DataSciences/QuantTrading/dataSet'
filePath = r'E:/Eskills-Academy-projects/tutorials/Masters-DataSciences/QuantTrading/dataSet/cm01JAN2018bhav.csv'

c.execute("Load data infile %s into table cmStaging fields terminated by ',' ignore 1 lines"
              "(symbol, series, open, high, low, close, last, prevclose, tottrdqty, tottrdval, @timestamp, totaltrades, isin)"
              "SET timestamp = STR_TO_DATE(@timestamp, '%d-%b-%Y')", filePath)

# import os
# for file in os.listdir(localExtractFilePath):
#     if file.endswith('.csv'):

#         fileName = localExtractFilePath+'/'+file

#         print(fileName)
#         c.execute("Load data infile %s into table cmStaging fields terminated by %s ignore 1 lines"
#               "(symbol, series, open, high, low, close, last, prevclose, tottrdqty, tottrdval, @timestamp, totaltrades, isin)"
#               "SET timestamp = STR_TO_DATE(@timestamp, %s)", (fileName, r',', r'%d-%b-%Y'))

#         # insertRows(localExtractFilePath+"/"+file, c)
#         print( "Loaded file", file, "into database")

