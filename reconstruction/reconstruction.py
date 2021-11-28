#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import time

import numpy as np
import pandas as pd

from anomaly.reconstruction import ReconstructionAnomalyScore

from autoencoders.deterministic.autoencoder import AE
from autoencoders.variational.autoencoder import VAE

from sdss.superclasses import FileDirectory

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("reconstruction.ini")
# Check files and directory
check = FileDirectory()
###############################################################################
# Load model
model_type = parser.get("common", "model_type")
model_directory = parser.get("directories", "model")

if model_type == "ae":

    model = AE.load(model_directory)

elif model_type == "vae":

    model = VAE.load(model_directory)

###############################################################################
# Load data
input_data_directory = parser.get("directories", "input")

train_set_name = parser.get("files", "observation")
observation = np.load(f"{input_data_directory}/{train_set_name}")

#######################################################################

reconstruction_name = parser.get("files", "reconstruction")
reconstruction_location = f"{model_directory}/{reconstruction_name}"

reconstruction_in_drive = parser.getboolean(
    "parameters", "reconstruction_in_drive"
)

if not reconstruction_in_drive:

    reconstruction = model.reconstruct(observation)
    np.save(reconstruction_location, reconstruction)

    reconstruction_in_drive = True  # to avoid recomputing it in .mse

else:

    reconstruction = np.load(reconstruction_location)
###############################################################################
# Load class to compute scores
analysis = ReconstructionAnomalyScore(model)
# metric parameters
metric = parser.get("score", "metric")
# relative = parser.getboolean("score", "relative")
relative_values = [True, False]

percentage = parser.get("score", "percentage")
percentage = percentage.replace(" ", "").split(",")
percentage_values = [int(value) for value in percentage]

# specobjid to save anomaly scores in data frame
train_id_name = parser.get("files", "train_id")
indexes_interpolate = np.load(f"{input_data_directory}/{train_id_name}")

succesful_interpolation = ~indexes_interpolate[:, 2].astype(np.bool)

specobjid = indexes_interpolate[succesful_interpolation, 1]
idx_train_set = indexes_interpolate[succesful_interpolation, 0]

###############################################################################
filter_narrow_lines = parser.getboolean("score", "filter_narrow_lines")
velocity_filter = parser.getfloat("score", "velocity_filter")
###############################################################################
data_frame = pd.DataFrame()

data_frame["specobjid"] = specobjid
data_frame["trainID"] = idx_train_set

for relative in relative_values:

    for percentage in percentage_values:

        anomaly_score = analysis.anomaly_score(
            metric=metric,
            observation=observation,
            percentage=percentage,
            relative=relative,
            filter_narrow_lines = filter_narrow_lines,
            velocity_filter = velocity_filter,
            reconstruction_in_drive=reconstruction_in_drive,
            reconstruction=reconstruction,
        )
        #######################################################################
        # Save anomaly scores
        output_directory = parser.get("directories", "output")
        check.check_directory(output_directory, exit=False)

        save_scores = parser.getboolean("parameters", "save_scores")

        if save_scores:

            anomaly_score_name = (
                f"{metric}_relative_{relative}_percentage_{percentage}"
            )

            save_to = f"{output_directory}/{anomaly_score_name}"

            np.save(f"{save_to}.npy", anomaly_score)
        #######################################################################
        # save to data frame
        data_frame["anomalyScore"] = anomaly_score

        data_frame.to_csv(f"{save_to}.csv.gz", index=False)
###############################################################################


finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
