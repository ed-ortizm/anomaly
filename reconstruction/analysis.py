#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np

from anomaly.analysis import AnalysisAnomalyScore
# from anomaly.reconstruction import ReconstructionAnomalyScore

from sdss.superclasses import FileDirectory

###############################################################################
start_time = time.time()
###########################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("analysis.ini")
# Check files and directory
check = FileDirectory()
###############################################################################
# Load data
input_data_directory = parser.get("directories", "input")
anomaly_score_name = parser.get("files", "anomaly_score")
anomaly_score = np.load(f"{input_data_directory}/{anomaly_score_name}")
###############################################################################
# Load class to analyse anomaies
analyze = AnalysisAnomalyScore(anomaly_score)
# get top scores
normal_number = parser.getint("parameters", "normal_number")
anomalous_number = parser.getint("parameters", "anomalous_number")

[
    normal_ids,
    anomalies_ids
] = analyze.top_scores(
    number_normal=normal_number,
    number_anomalies=anomalous_number
)

np.save(f"{input_data_directory}/normal_ids.npy", normal_ids)
np.save(f"{input_data_directory}/anomalous_ids.npy", anomalies_ids)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
