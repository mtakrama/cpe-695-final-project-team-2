#!/usr/bin/env python3

"""
Utilities to provide basic I/O and globally required global functions.
"""

import os
import csv
from enum import Enum


class FileType(Enum):
    CSV = 1
    XLS = 2
    XLSX = 3


def read(file_path, file_type=FileType.CSV):
    if file_path is None:
        raise Exception(
            "Filepath [{}] cannot be none (null)".format(file_path))
    if len(file_path) == 0:
        raise Exception("Filepath [{}] cannot be empty".format(file_path))
    if not os.path.exists(file_path):
        raise Exception("Filepath [{}] does not exist.".format(file_path))

    if file_type == FileType.CSV:
        return list(csv.reader(open(file_path)))

    raise Exception(
        "FileType [{}] for file [{}] unsupported".format(file_type, file_path))
