#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from anomaly.analysis import AnalysisAnomalyScore
from anomaly.reconstruction import ReconstructionAnomalyScore

from autoencoders.deterministic.autoencoder import AE
from autoencoders.variational.autoencoder import VAE

# from sdss.superclasses import FileDirectory

###############################################################################
if __name__ == "__main__":
    mp.set_start_method("spawn")

    start_time = time.time()
    ###########################################################################
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read("reconstruction.ini")
    # Check files and directory
    # check = FileDirectory()
    ###########################################################################
    # Load model
    model_type = parser.get("directories", "model_type")
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
    ###########################################################################
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
        )

    ###########################################################################
    finish_time = time.time()
    print(f"Run time: {finish_time - start_time:.2f}")
