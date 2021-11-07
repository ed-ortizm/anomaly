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
analyze_anomalies = AnalysisAnomalyScore(anomaly_score)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
