#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np
import pandas as pd

from anomaly.analysis import AnalysisAnomalyScore
from sdss.superclasses import FileDirectory, MetaData

###############################################################################
start_time = time.time()
###########################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("top_scores.ini")
# Check files and directory
check = FileDirectory()
###############################################################################
# Load data
input_output_directory = parser.get("directories", "input_output")
anomaly_score_name = parser.get("files", "anomaly_score")
anomaly_score = np.load(f"{input_output_directory}/{anomaly_score_name}")
###############################################################################
# Load class to analyse anomaies
analyze = AnalysisAnomalyScore(anomaly_score)
# get top scores
# normal_number = parser.getint("parameters", "normal_number")
# anomalous_number = parser.getint("parameters", "anomalous_number")

# [normal_train_idx, anomalous_train_idx] = analyze.top_scores(
#     number_normal=normal_number,
#     number_anomalies=anomalous_number
# )
normal_number = parser.getint("parameters", "normal_number")
normal_train_idx = analyze.top_scores(normal_number, anomalous=False)

anomalous_number = parser.getint("parameters", "anomalous_number")
anomalous_train_idx = analyze.top_scores(anomalous_number, anomalous=True)
###############################################################################
# Get specobjid from indexes_interpolate array
train_directory = parser.get("directories", "data")
train_id_name = parser.get("files", "train_id")
indexes_interpolate = np.load(f"{train_directory}/{train_id_name}")
#######################################################################
meta_data_directory = parser.get("directories", "meta_data")
spectra_df_name = parser.get("files", "spectra_df")
spectra_df_location = f"{meta_data_directory}/{spectra_df_name}"

spectra_df = pd.read_csv(spectra_df_location, index_col="specobjid")
#######################################################################
meta = MetaData()
# Get data frame for normal objects
# normal df. Add idTrain, score and url
specobjid_normal = indexes_interpolate[normal_train_idx][:, 1]
normal_df = spectra_df.loc[specobjid_normal]

normal_df["idTrain"] = normal_train_idx
# get scores
scores_normal = anomaly_score[normal_train_idx]
normal_df["anomalyScore"] = scores_normal
# Get urls in sdss dr16 object explorer
sdss_explorer_urls = []

for specobjid in specobjid_normal:

    url = meta.get_sky_server_url(specobjid)
    sdss_explorer_urls.append(url)

normal_df["explorerUrl"] = sdss_explorer_urls

#######################################################################
df_name = parser.get("files", "normal_df")
save_to = f"{input_output_directory}/{df_name}"
normal_df.to_csv(save_to, index=True)
###############################################################################
# Get data frame for normal objects
# normal df. Add idTrain, score and url
specobjid_anomalous = indexes_interpolate[anomalous_train_idx][:, 1]
anomalous_df = spectra_df.loc[specobjid_anomalous]

anomalous_df["idTrain"] = anomalous_train_idx
# get scores
scores_anomalous = anomaly_score[anomalous_train_idx]
anomalous_df["anomalyScore"] = scores_anomalous
# Get urls in sdss dr16 object explorer
sdss_explorer_urls = []

for specobjid in specobjid_anomalous:

    url = meta.get_sky_server_url(specobjid)
    sdss_explorer_urls.append(url)

anomalous_df["explorerUrl"] = sdss_explorer_urls

#######################################################################
df_name = parser.get("files", "anomalous_df")
save_to = f"{input_output_directory}/{df_name}"
anomalous_df.to_csv(save_to, index=True)
print(save_to)
###############################################################################
# print(analyze.get_percentiles())
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
