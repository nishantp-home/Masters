import findspark
findspark.init()

import pyspark

if __name__=='__main__':
    
    sc = pyspark.SparkContext("local[3]", "word count")    # Entry point to spark core, app name: word count
    sc.setLogLevel('ERROR')

    filePath = 'e:\\Eskills-Academy-projects\\python-spark-tutorial-master\\in\\'
    fileName = "word_count.text"
    textFile = filePath + fileName
    lines = sc.textFile(textFile)
    words = lines.flatMap(lambda line: line.split(" "))    # split article into separate words, using wide space as delimiter
    wordCounts = words.countByValue()
    for word, count in wordCounts.items():
        print("{}:{}".format(word, count))
