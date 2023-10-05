from urllib.request import urlopen
import json

url='https://www.data.qld.gov.au/api/3/action/datastore_search?resource_id=7afe7233-fae0-4024-bc98-3a72f05675bd&limit=5'
urlResult = urlopen(url)
rawData = urlResult.read()
jsonData = json.loads(rawData)

#jsonString = json.dumps(jsonData)
result = jsonData['result']
records = result['records']
firstRecord = records[0]
site = firstRecord['Site']
dateTime = firstRecord['DateTime']
waterLevel = firstRecord['Water Level']
prediction = firstRecord['Prediction']
residual = firstRecord['Residual']


fileName = 'StormTimeData.csv'
f =open(fileName, 'w')

headers = 'Site, DateTime, Water Level, Prediction, Residual \n'
f.write(headers)

for record in records:
    rowString = ''
    rowString += record['Site'] + ','
    rowString += record['DateTime'] + ','
    rowString += str(record['Water Level']) + ','
    rowString += str(record['Prediction']) + ','
    rowString += str(record['Residual']) + '\n'
    f.write(rowString)


