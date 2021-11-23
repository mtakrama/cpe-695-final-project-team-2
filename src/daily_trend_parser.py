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


def plot_all_daily_trends():
    print("Plotting daily trends...")
    # Discover case trend csv files and plot daily trends
    csv_files = glob.glob(data_prefix + '/data_table_for_daily_case_trends*')

    for csv_file in csv_files:
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

    plot_all_daily_trends()


main()
