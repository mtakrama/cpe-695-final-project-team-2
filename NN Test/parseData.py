import numpy as np
import csv
import time
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def readTrendDataFromCSV(file):
    # Read CSV and delete label rows/columns
    reader = csv.reader(open(file))
    np.nan_to_num(np.nan)
    data = list(reader)
    del data[0:3]
    data = np.array(data)
    data = np.delete(data, 0, 1)


    # Convert string dates to Unix timestamps for easy plotting
    for entry in data:
        eDateTime = datetime.strptime(entry[0], '%b %d %Y')
        entry[0] = int(time.mktime(eDateTime.timetuple()))
        if (entry[3] == ("N/A")):
            entry[3] = 0

    # Cast entire array to integer type
    return data.astype(np.float64)

plt.figure(0)
# Plot loaded trend data (x is Unix timestamp, Y is new case number)
alaskaCases = readTrendDataFromCSV('data_table_for_daily_case_trends__alaska.csv')
plt.plot(alaskaCases[:, 0], alaskaCases[:, 1], 'ro')
#print(alaskaCases)

# Create sets of 10 input data points and 1 output data point
# (model predicts the next case number given 10 previous case counts)
dataTimestamps = []
dataCuratedX = []
dataCuratedY = []
vaccinatedY = []

for index, entry in enumerate(alaskaCases):
    if index + 40 < len(alaskaCases):
        dataTimestamps.append(alaskaCases[index, 0])
        dataCuratedX.append(alaskaCases[index+1:index+41, 2])
        dataCuratedY.append(alaskaCases[index, 2])
        vaccinatedY.append(alaskaCases[index, 2])

dataCuratedX = np.array(dataCuratedX)
        
# Split training and test data
trainingSplitIndex = round(len(dataCuratedX) * 0.3)
dataTrainingTimestamps = dataTimestamps[trainingSplitIndex:]
dataTrainingX = np.asarray(dataCuratedX[trainingSplitIndex:])
dataTrainingY = np.asarray(dataCuratedY[trainingSplitIndex:])
vaccinatedTrainingY = np.asarray(vaccinatedY[trainingSplitIndex:])
dataTestTimestamps = dataTimestamps[:trainingSplitIndex]
dataTestX = np.asarray(dataCuratedX[:trainingSplitIndex])
dataTestY = np.asarray(dataCuratedY[:trainingSplitIndex])
vaccinatedTestY = np.asarray(vaccinatedY[:trainingSplitIndex])


