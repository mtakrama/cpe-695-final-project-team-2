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

import utilities as util

data_prefix = os.path.join(os.path.dirname(__file__), '../data')
plot_path = os.path.join(os.path.dirname(__file__), 'plots')

xaxis_plotter = []
filename_plotter = []

numPastDays = 40
numFutureDays = 20


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

    # setup daily trend path
    daily_trend_path = os.path.join(plot_path, 'daily_trends_raw')
    os.mkdir(daily_trend_path)

    for csv_file in data_files:
        daily_cases = read(os.path.join(data_prefix, csv_file))
        xaxis_plotter.append(len(daily_cases))

        # grab the state name from file path
        filename = re.search('.*__(.+?)\.csv', csv_file).group(1)
        filename_plotter.append(filename)

        plot(filename=os.path.join(daily_trend_path, filename + "_raw_daily_case_trends.png"),
             title="{} Daily Case Trend".format(filename.title()),
             xlabel="Unix Time Stamp",
             ylabel="Number of Cases",
             xdata=daily_cases[:, 0],
             ydata=daily_cases[:, 2])

    print("Done plotting daily trends.")


def prepare_data(csv_files):
    # These arrays contain the per-day counts for number of cases and new vaccinations,
    # and associated Unix timestamps
    dataTimestamps = []
    dataRawCases = []
    dataRawVaccinated = []

    # This array contains smaller arrays that have been created as a working set of
    # inputs data points for the first layer of the network
    dataCuratedX = []

    # This array contains smaller arrays that are the expected outputs for the input
    # data points given by 'dataCuratedX'
    dataCuratedY = []

    for csv_file in csv_files:
        daily_cases = read(os.path.join(data_prefix, csv_file))

        for index, entry in enumerate(daily_cases):
            if index > numFutureDays and index + numPastDays < len(daily_cases):
                dataTimestamps.append(daily_cases[index, 0])
                dataRawCases.append(daily_cases[index, 2])
                dataRawVaccinated.append(daily_cases[index, 3])

                dataInputX = []
                dataInputX.extend(daily_cases[index+1:index+numPastDays+1, 2])
                dataInputX.extend(daily_cases[index+1:index+numPastDays+1, 3])
                dataCuratedX.append(dataInputX[:])
                dataCuratedY.append(
                    daily_cases[index-numFutureDays+1:index+1, 2])

    dataCuratedX = np.array(dataCuratedX)

    # Split training and test data
    trainingSplitIndex = round(len(dataCuratedX) * 0.3)

    dataTrainingTimestamps = dataTimestamps[trainingSplitIndex:]
    dataTrainingX = np.asarray(dataCuratedX[trainingSplitIndex:])
    dataTrainingY = np.asarray(dataCuratedY[trainingSplitIndex:])

    dataTestTimestamps = dataTimestamps[:trainingSplitIndex]
    dataTestX = np.asarray(dataCuratedX[:trainingSplitIndex])
    dataTestY = np.asarray(dataCuratedY[:trainingSplitIndex])

    return dataTimestamps, dataRawCases, dataRawVaccinated, (dataTrainingTimestamps, dataTrainingX, dataTrainingY), (dataTestTimestamps, dataTestX, dataTestY)


def define_compile_model(optimizer_str, loss_str, metrics_list):
    # Build and train model
    model = keras.Sequential()
    model.add(layers.Dense(numPastDays * 2, activation="relu"))
    model.add(layers.Dense(30, activation="relu"))
    model.add(layers.Dense(numFutureDays))

    model.compile(
        optimizer=optimizer_str,
        loss=loss_str,
        metrics=metrics_list
    )

    return model


def plot_loss(history, epochs):
    # plot loss
    hLoss = history.history['loss']
    hVLoss = history.history['val_loss']
    plt.figure()
    plt.plot(range(epochs), hLoss, '-', range(epochs), hVLoss, '--')
    plt.savefig(os.path.join(plot_path, "loss_plot.png"))
    plt.close()


def main():
    setup_plot_paths()

    # Discover case trend csv files and plot daily trends
    csv_files = glob.glob(data_prefix + '/data_table_for_daily_case_trends*')

    plot_all_daily_trends(csv_files)

    # preparation
    training_data_tpl = tuple()
    test_data_tpl = tuple()
    dataTimestamps, dataRawCases, dataRawVaccinated, training_data_tpl, test_data_tpl = prepare_data(
        csv_files)

    # model definition
    model = define_compile_model(
        'adam', 'mean_squared_error', ['accuracy'])

    epochs = 200
    history = model.fit(training_data_tpl[1], training_data_tpl[2],
                        epochs=epochs,
                        validation_data=(test_data_tpl[1], test_data_tpl[2]),
                        verbose=0
                        )

    # model evaluation
    eval_info = model.evaluate(test_data_tpl[1], test_data_tpl[2])
    print('Loss: {}, Accuracy: {}'.format(eval_info[0], eval_info[1]))

    plot_loss(history, epochs)

    prettyplotter(dataRawCases, dataRawVaccinated, test_data_tpl[1], test_data_tpl[2], training_data_tpl[1], training_data_tpl[2],
                  training_data_tpl[0], test_data_tpl[0], dataTimestamps, model, numPastDays, numFutureDays)


def prettyplotter(dataRawCases, dataRawVaccinated, dataTestX, dataTestY, dataTrainingX, dataTrainingY, dataTrainingTimestamps, dataTestTimestamps, dataTimestamps, model, numPastDays, numFutureDays):

    plt.figure()
    print('Pretty Model vs training data (error)')
    i = 0
    accum = numFutureDays
    for xaxisTrainPlot in xaxis_plotter:
        if i < 10:  # plot first 10
            plt.figure()

            dataModelInput = []
            dataModelInput.extend(dataRawCases[accum:accum+numPastDays])
            dataModelInput.extend(dataRawVaccinated[accum:accum+numPastDays])
            dataTrainingPredY = model.predict(
                np.asarray([dataModelInput[:]]))[0]

            plt.plot(dataTimestamps[accum-numFutureDays:accum+numPastDays],
                     dataRawCases[accum-numFutureDays:accum+numPastDays], label='Input Case Data')
            plt.plot(dataTimestamps[accum-numFutureDays:accum+numPastDays],
                     dataRawVaccinated[accum-numFutureDays:accum+numPastDays], label='Input Vaccinated Data')
            plt.plot(dataTimestamps[accum-numFutureDays:accum],
                     dataTrainingPredY, label='Predicted Data')
            plt.title("{} Model vs Training Data ".format(
                filename_plotter[i].title()))
            plt.xlabel("Unix Time Stamp")
            plt.ylabel("Number of Cases")
            plt.legend(loc="upper right")

            accum = accum + xaxisTrainPlot
            figtitle = "pretty_model_vs_training_data_error_" + \
                str(filename_plotter[i]) + ".png"
            i = i + 1

            plt.savefig(os.path.join(
                plot_path, figtitle))
            plt.close()


main()
