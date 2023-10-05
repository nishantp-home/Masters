# Script to download and unzip all the files from 2016 onwards

from download import *
import numpy as np

listOfMonths = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
startYear = 2016
endYear = 2023
listOfYearsLength = endYear - startYear + 1
listOfYears = np.linspace(startYear, endYear, listOfYearsLength)

for year in listOfYears:
    for month in listOfMonths:
        for dayOfMonth in range(31):

            day = dayOfMonth + 1    # 
