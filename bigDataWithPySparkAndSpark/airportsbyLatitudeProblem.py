import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from Utils import Utils


# if __name__ == "__main__":

'''
Create a Spark program to read the airport data from in/airports.text,  find all the airports whose latitude are bigger than 40.
Then output the airport's name and the airport's latitude to out/airports_by_latitude.text.

Each row of the input file contains the following columns:
Airport ID, Name of airport, Main city served by airport, Country where airport is located, IATA/FAA code,
ICAO Code, Latitude, Longitude, Altitude, Timezone, DST, Timezone in Olson format

Sample output:
"St Anthony", 51.391944
"Tofino", 49.082222
...
'''
latitudeIndex = 6

def splitComma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[latitudeIndex])

conf = SparkConf().setAppName('airports').setMaster('local[2]')
sc = SparkContext(conf=conf)


filePath = 'e:\\Eskills-Academy-projects\\python-spark-tutorial-master\\in\\'
fileName = "airports.text"
outFolder = "airportsByLatitude"
textFile = filePath + fileName
airports = sc.textFile(textFile)

minLatitude = 40.
airportsSatifyingLatitudeConstaint = airports.filter(lambda line: float(Utils.COMMA_DELIMITER.split(line)[latitudeIndex]) >= minLatitude)
airportNameAndCityNames = airportsSatifyingLatitudeConstaint.map(splitComma)

# store the output folder in the folder containing this python file

airportNameAndCityNames.saveAsTextFile(outFolder)

