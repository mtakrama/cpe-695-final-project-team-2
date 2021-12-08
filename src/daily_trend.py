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


def convert_to_unix(date_str):
    eDateTime = datetime.strptime(date_str, '%b %d %Y')
    return int(time.mktime(eDateTime.timetuple()))


def zeroize_invalid_vaccination_rate(vaccine_rate_str):
    if vaccine_rate_str == "N/A":
        return 0
    return vaccine_rate_str


def read(file):
    try:
        data = util.read(file, util.FileType.CSV)
        # delete first 3 lines
        del data[0:3]

        # blow out first column
        data = np.array(data)
        data = np.delete(data, 0, 1)

        # Clean data
        for entry in data:
            # print('Entry: {}'.format(entry))
            entry[0] = convert_to_unix(entry[0])
            entry[3] = zeroize_invalid_vaccination_rate(entry[3])

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


def grab_state_name(path_str):
    return re.search('.*__(.+?)\.csv', path_str).group(1)


def plot_all_daily_trends(data_files):
    print("Plotting daily trends...")

    # setup daily trend path
    daily_trend_path = os.path.join(plot_path, 'daily_trends_raw')
    os.mkdir(daily_trend_path)

    for csv_file in data_files:
        daily_cases = read(os.path.join(data_prefix, csv_file))
        xaxis_plotter.append(len(daily_cases))

        # grab the state name from file path
        filename = grab_state_name(csv_file)
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

    # State level validation data
    state_level_validation_data_lst = []

    for csv_file in csv_files:
        daily_cases = read(os.path.join(data_prefix, csv_file))

        current_state = grab_state_name(csv_file)
        # print('Extracting data for {} state...'.format(current_state))

        state_data_x = []
        state_data_y = []
        for index, entry in enumerate(daily_cases):
            if index > numFutureDays and index + numPastDays < len(daily_cases):
                dataTimestamps.append(daily_cases[index, 0])
                dataRawCases.append(daily_cases[index, 2])
                dataRawVaccinated.append(daily_cases[index, 3])

                dataInputX = []
                dataInputX.extend(
                    daily_cases[index+1:index+numPastDays+1, 2])
                dataInputX.extend(
                    daily_cases[index+1:index+numPastDays+1, 3])

                data_input_x = dataInputX[:]
                data_input_y = daily_cases[index-numFutureDays+1:index+1, 2]

                dataCuratedX.append(data_input_x)
                dataCuratedY.append(data_input_y)

                # state level raw data
                state_data_x.append(data_input_x)
                state_data_y.append(data_input_y)

        state_level_split = round(len(state_data_x) * 0.3)
        state_level_validation_x = state_data_x[:state_level_split]
        state_level_validation_y = state_data_y[:state_level_split]

        state_level_validation_data_lst.append(
            (current_state, np.array(state_level_validation_x), np.array(state_level_validation_y)))

        # print(state_level_validation_data_lst[0][2])

        # inspect data
        # for state_validation_data in state_level_validation_data_lst:
        #     print(state_validation_data)
        #     print('Timestamp length: {}'.format(len(state_validation_data[1])))

    dataCuratedX = np.array(dataCuratedX)

    # Split training and test data
    trainingSplitIndex = round(len(dataCuratedX) * 0.3)

    dataTrainingTimestamps = dataTimestamps[trainingSplitIndex:]
    dataTrainingX = np.asarray(dataCuratedX[trainingSplitIndex:])
    dataTrainingY = np.asarray(dataCuratedY[trainingSplitIndex:])

    dataTestTimestamps = dataTimestamps[:trainingSplitIndex]
    dataTestX = np.asarray(dataCuratedX[:trainingSplitIndex])
    dataTestY = np.asarray(dataCuratedY[:trainingSplitIndex])

    return dataTimestamps, dataRawCases, dataRawVaccinated, (dataTrainingTimestamps, dataTrainingX, dataTrainingY), (dataTestTimestamps, dataTestX, dataTestY), state_level_validation_data_lst


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


def evaluate_loss_per_state(model, state_level_validation_data_lst):
    # model evaluation
    state_mse_pairs = []
    for state_validation_data in state_level_validation_data_lst:
        state_name = state_validation_data[0]

        if state_name not in ['idaho', 'mississippi', 'washington', 'delaware']:
            continue

        eval_info = model.evaluate(
            state_validation_data[1], state_validation_data[2])

        state_name = state_validation_data[0]
        loss_metric = eval_info[0]
        mse_metric = eval_info[1]

        print('-'*100)
        print('State: {}, MSE metric: {}'.format(
            state_name, np.sqrt(mse_metric)))

        state_mse_pairs.append((state_name, mse_metric))

    # plot bar chart
    states = [state_mse_pair[0] for state_mse_pair in state_mse_pairs]
    rmses = np.sqrt([state_mse_pair[1] for state_mse_pair in state_mse_pairs])

    # creating the bar plot
    plt.figure()
    plt.bar(states, rmses, color='maroon',
            width=0.4)
    plt.xlabel("States")
    plt.ylabel("Root Mean Square Error")
    plt.title("RMSE per state")
    plt.savefig(os.path.join(plot_path, 'mse_by_state.png'))
    plt.close()


def main():
    setup_plot_paths()

    # Discover case trend csv files and plot daily trends
    csv_files = glob.glob(data_prefix + '/data_table_for_daily_case_trends*')

    plot_all_daily_trends(csv_files)

    # preparation
    training_data_tpl = tuple()
    test_data_tpl = tuple()
    dataTimestamps, dataRawCases, dataRawVaccinated, training_data_tpl, test_data_tpl, state_level_validation_data_lst = prepare_data(
        csv_files)

    # model definition
    model = define_compile_model(
        'adam', 'mean_squared_error', ['mse'])

    epochs = 200
    history = model.fit(training_data_tpl[1], training_data_tpl[2],
                        epochs=epochs,
                        validation_data=(test_data_tpl[1], test_data_tpl[2]),
                        verbose=0
                        )

    evaluate_loss_per_state(model, state_level_validation_data_lst)

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
