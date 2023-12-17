import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from Utils import Utils

def splitComma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[2])

conf = SparkConf().setAppName('airports').setMaster('local[2]')
sc = SparkContext(conf=conf)


filePath = 'e:\\Eskills-Academy-projects\\python-spark-tutorial-master\\in\\'
fileName = "airports.text"
outFolder = "airportsInUSA"
textFile = filePath + fileName
airports = sc.textFile(textFile)

airportsInUSA = airports.filter(lambda line: Utils.COMMA_DELIMITER.split(line)[3] == "\"United States\"")
airportNameAndCityNames = airportsInUSA.map(splitComma)

# store the output folder in the folder containing this python file
airportNameAndCityNames.saveAsTextFile(outFolder)

