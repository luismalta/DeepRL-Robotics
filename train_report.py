from statistics import mean, median,pstdev
import os.path
import os
import errno
import argparse
import sys

import json
import numpy as np

from datetime import datetime

parser = argparse.ArgumentParser(description='Train report build')
parser.add_argument('-a', '--history', help='History file name',default=None)
args = parser.parse_args()


def build_test_report(train_history):
    report = {
        "Mean" :  mean(train_history["episode_reward"]),
        "Median" : median(train_history["episode_reward"]),
        "Standard deviation" : pstdev(train_history["episode_reward"])

    }

    with open(args.history[:-18] + '/train_report.json', 'w') as fp:
        json.dump(report, fp)

with open(args.history, 'r') as fp:
    train_history = json.load(fp)
    build_test_report(train_history)
