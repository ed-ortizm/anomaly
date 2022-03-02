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

import numpy as np
import pandas as pd
import tensorflow as tf

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
# Load data
print("Load observations")
data_directory = parser.get("directory", "input")

observation_name = parser.get("file", "observation")
observation = np.load(f"{data_directory}/{observation_name}")
meta_data_directory = parser.get("directory", "meta_data")
wave_name = parser.get("file", "grid")
wave = np.load(f"{meta_data_directory}/{wave_name}")
###############################################################################
# set the number of cores to use per model in each worker
jobs = parser.getint("tf-session", "cores")
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=jobs,
    inter_op_parallelism_threads=jobs,
    allow_soft_placement=True,
    device_count={"CPU": jobs},
)
session = tf.compat.v1.Session(config=config)
# Load reconstruction function
print(f"Load reconstruction function", end="\n")

model_directory = parser.get("directory", "model")
model = parser.get("file", "model_id")
model_location = f"{model_directory}/{model}"
model = AutoEncoder(reload=True, reload_from=model_location)
# reconstruct_function = model.reconstruct
#
# score_config = parser.items("score")
# score_config = configuration.section_to_dictionary(score_config, [",", "\n"])
#
# save_to = parser.get("directory", "output")
# check.check_directory(save_to, exit=False)
#
# ###############################################################################
# # specobjid to save anomaly scores in data frame
# print("Track meta data", end="\n")
#
# idx_specobjid_name = parser.get("file", "specobjid")
# idx_specobjid = np.load(f"{data_directory}/{idx_specobjid_name}")
#
# specobjid = idx_specobjid[:, 1]
# idx_train_set = idx_specobjid[:, 0]
#
# data_frame = pd.DataFrame()
#
# data_frame["specobjid"] = specobjid
# data_frame["trainID"] = idx_train_set
# ###############################################################################
# save_score = parser.getboolean("score", "save_score")
# for metric in score_config["metric"]:
#
#     for filter in score_config["filter"]:
#
#         for percentage in score_config["percentage"]:
#
#             for relative in score_config["relative"]:
#
#                 if filter is False:
#
#                     ###########################################################
#                     anomaly = ReconstructionAnomalyScore(
#                         reconstruct_function,
#                         wave,
#                         lines=None,
#                         percentage=percentage,
#                         relative=relative,
#                         epsilon=1e-3,
#                     )
#                     ###########################################################
#                     score_name = f"{metric}_percent_{percentage}"
#
#                     if relative is True:
#                         score_name = f"{score_name}_relative"
#
#                     print(f"Score: {score_name}", end="\r")
#
#                     score = anomaly.score(observation, metric)
#
#                     data_frame[f"{score_name}"] = score
#
#                     if save_score is True:
#                         np.save(f"{save_to}/{score_name}.npy", score)
#                 else:
#
#                     for velocity in score_config["velocity"]:
#
#                         #######################################################
#                         anomaly = ReconstructionAnomalyScore(
#                             reconstruct_function,
#                             wave,
#                             lines=score_config["lines"],
#                             velocity_filter=velocity,
#                             percentage=percentage,
#                             relative=relative,
#                             epsilon=1e-3,
#                         )
#                         #######################################################
#                         score_name = ( f"{metric}_percent_{percentage}_filter_{velocity}kms"
#                         )
#
#                         if relative is True:
#                             score_name = f"{score_name}_relative"
#
#                         print(f"Score: {score_name}", end="\r")
#
#                         score = anomaly.score(observation, metric)
#                         data_frame[f"{score_name}"] = score
#
#                         if save_score is True:
#                             np.save(f"{save_to}/{score_name}.npy", score)
# ###############################################################################
# # save to data frame
# scores_frame_name = parser.get("file", "scores_frame")
# data_frame.to_csv(f"{save_to}/{scores_frame_name}", index=False)
###############################################################################
session.close()
finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
