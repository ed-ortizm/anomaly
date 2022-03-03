import os

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
# Set TensorFlow print of log information
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
###############################################################################
from configparser import ConfigParser, ExtendedInterpolation
import sys
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
import pandas as pd
import tensorflow as tf

from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly import parallelReconstruction
from autoencoders.ae import AutoEncoder
from sdss.superclasses import FileDirectory, ConfigurationFile
###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    ###########################################################################
    start_time = time.time()
    ###########################################################################
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read("reconstruction.ini")
    # Check files and directory
    check = FileDirectory()
    # Handle configuration file
    configuration = ConfigurationFile()
    ###########################################################################
    # Load data
    print("Load observations")

    ###########################################################################
    counter = mp.Value("i", 0)

    ###########################################################################
    data_directory = parser.get("directory", "input")
    observation_name = parser.get("file", "observation")
    observation = np.load(f"{data_directory}/{observation_name}")
    share_observation = RawArray(
        np.ctypeslib.as_ctypes_type(observatio.dtype), observation
    )

    observation_shape = observation.shape
    del observation

    ###########################################################################
    meta_data_directory = parser.get("directory", "meta_data")
    wave_name = parser.get("file", "grid")
    wave = np.load(f"{meta_data_directory}/{wave_name}")
    share_wave = RawArray(
        np.ctypeslib.as_ctypes_type(wave.dtype), wave
    )

    del wave

    ###########################################################################
    print("Track meta data", end="\n")

    specobj_ids_name = parser.get("file", "specobjid")
    specobj_ids = np.load(f"{data_directory}/{specobj_ids_name}")

    specobj_id = idx_specobjid[:, 1]
    share_specobj_id = RawArray(
        np.ctypeslib.as_ctypes_type(share_specobj_id.dtype), share_specobj_id
    )
    del specobj_id

    train_id = specobj_ids[:, 0]
    share_train_id = RawArray(
        np.ctypeslib.as_ctypes_type(train_id.dtype), train_id
    )
    del train_id

    ###########################################################################
    model_directory = parser.get("directory", "model")
    model = parser.get("file", "model_id")
    share_model_directory = f"{model_directory}/{model}"
    check.check_directory(share_model_directory, exit=True)

    share_output_directory = parser.get("directory", "output")
    check.check_directory(share_output_directory, exit=False)
    ###########################################################################
    # Define grid for anomaly score function
    score_config = parser.items("score")
    score_config = configuration.section_to_dictionary(
        score_config, [",", "\n"]
    )
    parameters_grid = parallelReconstruction.get_grid(score_config)
    ###########################################################################
    number_processes = parser.getint("configuration", "jobs")
    cores_per_worker = parser.getint("configuration", "cores_per_worker")

    with mp.Pool(
        processes=number_processes,
        initializer=parallelReconstruction.init_shared_data,
        initargs=(
            counter,
            share_wave,
            share_observation,
            observation_shape,
            share_specobj_id,
            share_train_id,
            shared_model_directory,
            share_output_directory,
            cores_per_worker,
        ),
    ) as pool:

    pool.starmap(parallelReconstruction.compute_anomaly_score, grid)
    ###########################################################################
    save_score = parser.getboolean("score", "save_score")
    for metric in score_config["metric"]:

        for filter in score_config["filter"]:

            for percentage in score_config["percentage"]:

                for relative in score_config["relative"]:

                    if filter is False:

                        #######################################################
                        anomaly = ReconstructionAnomalyScore(
                            reconstruct_function,
                            wave,
                            lines=None,
                            percentage=percentage,
                            relative=relative,
                            epsilon=1e-3,
                        )
                        ###########################################################
                        score_name = f"{metric}_percent_{percentage}"

                        if relative is True:
                            score_name = f"{score_name}_relative"

                        print(f"Score: {score_name}", end="\r")

                        score = anomaly.score(observation, metric)

                        data_frame[f"{score_name}"] = score

                        if save_score is True:
                            np.save(f"{save_to}/{score_name}.npy", score)
                    else:

                        for velocity in score_config["velocity"]:

                            #######################################################
                            anomaly = ReconstructionAnomalyScore(
                                reconstruct_function,
                                wave,
                                lines=score_config["lines"],
                                velocity_filter=velocity,
                                percentage=percentage,
                                relative=relative,
                                epsilon=1e-3,
                            )
                            #######################################################
                            score_name = ( f"{metric}_percent_{percentage}_filter_{velocity}kms"
                            )

                            if relative is True:
                                score_name = f"{score_name}_relative"

                            print(f"Score: {score_name}", end="\r")

                            score = anomaly.score(observation, metric)
                            data_frame[f"{score_name}"] = score

                            if save_score is True:
                                np.save(f"{save_to}/{score_name}.npy", score)
    ###############################################################################
    # save to data frame
    scores_frame_name = parser.get("file", "scores_frame")
    data_frame.to_csv(f"{save_to}/{scores_frame_name}", index=False)
    ###############################################################################
    session.close()
    finish_time = time.time()
    print(f"Run time: {finish_time - start_time:.2f}")
