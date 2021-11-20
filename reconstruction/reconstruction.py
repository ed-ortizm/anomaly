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
###########################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("reconstruction.ini")
# Check files and directory
check = FileDirectory()
###########################################################################
# Load model
model_type = parser.get("common", "model_type")
model_directory = parser.get("directories", "model")


if model_type == "ae":

    model = AE.load(model_directory)

elif model_type == "vae":

    model = VAE.load(model_directory)

###########################################################################
# Load data
input_data_directory = parser.get("directories", "input")
train_set_name = parser.get("files", "observations")
observations = np.load(f"{input_data_directory}/{train_set_name}")
###############################################################################
reconstructions_name = parser.get("files", "reconstructions")
reconstruction_location = f"{model_directory}/{reconstructions_name}"

reconstruction_in_drive = parser.getboolean(
    "parameters", "reconstruction_in_drive"
    )

if not reconstruction_in_drive:

    reconstruction = model.reconstruct(observations)

    # reconstructions_name = parser.get("files", "reconstructions")
    # save_to = f"{model_directory}/{reconstructions_name}"

    np.save(reconstruction_location, reconstruction)

else:
    reconstruction = np.load(reconstruction_location)
###############################################################################
# Load class to compute scores
analyze_anomalies = ReconstructionAnomalyScore(model)
# metric parameters
metric = parser.get("score", "metric")
relative = parser.getboolean("score", "relative")
percentage = parser.getfloat("score", "percentage")

if metric == "mse":

    anomaly_score = analyze_anomalies.mse(
        observation=observations,
        percentage=percentage,
        relative=relative,
        reconstruction_in_drive = reconstruction_in_drive,
        reconstruction = reconstruction
    )
###############################################################################
output_directory = parser.get("directories", "output")
check.check_directory(output_directory, exit=False)

save_scores = parser.getboolean("parameters", "save_scores")

if save_scores:

    anomaly_score_name = parser.get("files", "anomaly_score")
    save_to = f"{output_directory}/{anomaly_score_name}"

    np.save(save_to, anomaly_score)
    print(save_to)
###########################################################################
finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
