#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import sys
import time

import numpy as np
import pandas as pd

from anomaly.reconstruction import ReconstructionAnomalyScore

from autoencoders.ae import AutoEncoder

from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("reconstruction.ini")
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
###############################################################################
# Load model
model_directory = parser.get("directories", "model")
model = AutoEncoder(reload=True, reload_from=model_directory)

###############################################################################
# Load data
print("Load observations")
input_data_directory = parser.get("directories", "input")

train_set_name = parser.get("files", "observation")
observation = np.load(f"{input_data_directory}/{train_set_name}")
#######################################################################
reconstruction_name = parser.get("files", "reconstruction")
reconstruction_location = f"{model_directory}/{reconstruction_name}"

print("Load reconstructions")
reconstruction_in_drive = parser.getboolean(
    "parameters", "reconstruction_in_drive"
)

if reconstruction_in_drive is False:

    reconstruction = model.reconstruct(observation)
    np.save(reconstruction_location, reconstruction)

    reconstruction_in_drive = True  # to avoid recomputing it in .mse

else:

    reconstruction = np.load(reconstruction_location)
###############################################################################
# Load class to compute scores
print("Load anomaly analysis class")
wave_name = parser.get("files", "grid")
wave = np.load(f"{input_data_directory}/{wave_name}")

analysis = ReconstructionAnomalyScore(model, wave)
###############################################################################
# metric parameters
print("Set parameters of metrics")
score_items = parser.items("score")
score_parameters = configuration.section_to_dictionary(score_items, [","])

metric = score_parameters["metric"]

relative_values = score_parameters["relative"]

percentage_values = score_parameters["percentage"]
#######################################################################
lines_items = parser.items("lines")
lines_parameters = configuration.section_to_dictionary(lines_items, [])

lines = lines_parameters["lines"]
filter_lines = lines_parameters["filter"]
velocity_filter = lines_parameters["velocity"]
###############################################################################
# specobjid to save anomaly scores in data frame
print("Set meta data tracking")
train_id_name = parser.get("files", "train_id")
indexes_interpolate = np.load(f"{input_data_directory}/{train_id_name}")

succesful_interpolation = ~indexes_interpolate[:, 2].astype(bool)

specobjid = indexes_interpolate[succesful_interpolation, 1]
idx_train_set = indexes_interpolate[succesful_interpolation, 0]

data_frame = pd.DataFrame()

data_frame["specobjid"] = specobjid
data_frame["trainID"] = idx_train_set
###############################################################################
save_scores = parser.getboolean("parameters", "save_scores")
output_directory = parser.get("directories", "output")
output_directory = f"{output_directory}/{metric}"
check.check_directory(output_directory, exit=False)

for relative in relative_values:

    for percentage in percentage_values:

        print(
            f"Filter:{filter_lines},Relative:{relative}, {percentage}%",
            end="\n",
        )

        anomaly_score = analysis.anomaly_score(
            metric=metric,
            observation=observation,
            percentage=percentage,
            relative=relative,
            filter_lines=filter_lines,
            lines=lines,
            velocity_filter=velocity_filter,
            reconstruction_in_drive=reconstruction_in_drive,
            reconstruction=reconstruction,
        )
        #######################################################################
        # Save anomaly scores
        score_name = (
            f"{metric}_relative_{relative}_percentage_{percentage}"
            f"_filter_{filter_lines}"
        )

        if filter_lines is True:
            score_name = f"{score_name}_{velocity_filter}kms"

        save_to = f"{output_directory}/{score_name}"

        # save to data frame
        data_frame["anomalyScore"] = anomaly_score

        data_frame.to_csv(f"{save_to}.csv.gz", index=False)

        #######################################################################
        if save_scores:

            np.save(f"{save_to}.npy", anomaly_score)
###############################################################################


finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
