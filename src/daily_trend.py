#!/usr/bin/env python3

"""
Daily case trend dedicated parser.
"""

import shutil
import re
import random
import glob
import os
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

import utilities as util

data_prefix = os.path.join(os.path.dirname(__file__), '../data')
plot_path = os.path.join(os.path.dirname(__file__), 'plots')


def read(file):
    try:
        data = util.read(file, util.FileType.CSV)
        # delete first 3 lines
        del data[0:3]

        # blow out first column
        data = np.array(data)
        data = np.delete(data, 0, 1)

        # Convert string dates to Unix timestamps for easy plotting
        for entry in data:
            eDateTime = datetime.strptime(entry[0], '%b %d %Y')
            entry[0] = int(time.mktime(eDateTime.timetuple()))
            if (entry[3] == ("N/A")):
                entry[3] = 0
    except Exception as e:
        print(e)
        return None

    # Cast entire array to integer type
    return data.astype(np.float64)


def plot(filename, title, xlabel, ylabel, xdata, ydata):
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # temporily gen plot color on the fl
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)

    plt.plot(xdata, ydata, c=color)

    print(" - saving " + filename)
    plt.savefig(filename)
    plt.close()


def setup_plot_paths():
    # Cleanup and setup plots path
    if os.path.exists(os.path.join(os.getcwd(), plot_path)):
        shutil.rmtree(plot_path, ignore_errors=True)
    os.mkdir(plot_path)


def plot_all_daily_trends(data_files):
    print("Plotting daily trends...")

    for csv_file in data_files:
        daily_cases = read(os.path.join(data_prefix, csv_file))

        # grab the state name from file path
        filename = re.search('.*__(.+?)\.csv', csv_file).group(1)

        plot(filename=os.path.join(plot_path, filename + "_daily_case_trends" + '.png'),
             title="{} Daily Case Trend".format(filename.title()),
             xlabel="Unix Time Stamp",
             ylabel="Number of Cases",
             xdata=daily_cases[:, 0],
             ydata=daily_cases[:, 2])

    print("Done plotting daily trends.")


def main():
    setup_plot_paths()

    # Discover case trend csv files and plot daily trends
    csv_files = glob.glob(data_prefix + '/data_table_for_daily_case_trends*')

    plot_all_daily_trends(csv_files)

    # Create sets of 10 input data points and 1 output data point
    # (model predicts the next case number given 10 previous case counts)
    dataTimestamps = []
    dataCuratedX = []
    dataCuratedY = []
    vaccinatedY = []

    for csv_file in csv_files:
        # TODO: inneficient, change to read only once.
        daily_cases = read(os.path.join(data_prefix, csv_file))

        for index, entry in enumerate(daily_cases):
            if index + 40 < len(daily_cases):
                dataTimestamps.append(daily_cases[index, 0])
                dataCuratedX.append(daily_cases[index+1:index+41, 2])
                dataCuratedY.append(daily_cases[index, 2])
                vaccinatedY.append(daily_cases[index, 3])

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

    # Build and train model
    model = keras.Sequential()
    model.add(layers.Dense(40, activation="relu"))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(5, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='mean_squared_error'
    )

    epochs = 200

    history = model.fit(dataTrainingX, dataTrainingY,
                        epochs=epochs,
                        validation_data=(dataTestX, dataTestY),
                        verbose=0
                        )

    # Plot loss
    hLoss = history.history['loss']
    hVLoss = history.history['val_loss']
    plt.figure()
    plt.plot(range(epochs), hLoss, '-', range(epochs), hVLoss, '--')
    plt.savefig(os.path.join(plot_path, "loss_plot.png"))
    plt.close()

    print('Model vs training data (error)')
    dataTrainingPredY = model.predict(dataTrainingX)
    plt.figure()
    plt.plot(dataTrainingTimestamps, dataTrainingY, '-',
             dataTrainingTimestamps, dataTrainingPredY, '--')
    plt.savefig(os.path.join(
        plot_path, "model_vs_training_data_error.png"))
    # print("accuracy_score", accuracy_score(
    #     dataTrainingY, dataTrainingPredY))
    plt.close()

    print('Model vs test data (error)')
    dataTestPredY = model.predict(dataTestX)
    plt.figure()
    plt.plot(dataTestTimestamps, dataTestY, '-',
             dataTestTimestamps, dataTestPredY, '--')
    plt.savefig(os.path.join(plot_path, "model_vs_test_data.png"))
    plt.close()
    # print("accuracy_score", accuracy_score(dataTestY, dataTestPredY))


main()
