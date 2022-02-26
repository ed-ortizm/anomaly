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
###############################################################################
# Load data
print("Load observations")
data_directory = parser.get("directory", "input")

observation_name = parser.get("files", "observation")
observation = np.load(f"{data_directory}/{observation_name}")
meta_data_directory = parser.get("directory", "meta_data")
wave_name = parser.get("files", "grid")
wave = np.load(f"{meta_data_directory}/{wave_name}")
###############################################################################
# Load reconstruction function
print(f"Load reconstruction function", end="\n")

model_directory = parser.get("directory", "model")
model = AutoEncoder(reload=True, reload_from=model_directory)
reconstruct_function = model.reconstruct

score_config = parser.items("score")
score_config = configuration.section_to_dictionary(score_config, [",", "\n"])

save_to = parser.get("directory", "output")
check.check_directory(save_to, exit=False)

for metric in score_config["metric"]:

    for filter in score_config["filter"]:

        for percentage in score_config["percentage"]:

            for relative in score_config["relative"]:

                if filter is False:

                    ###########################################################
                    anomaly = ReconstructionAnomalyScore(
                        reconstruct_function,
                        wave,
                        lines=None,
                        percentage=percentage,
                        relative=relative,
                        epsilon=1e-3,
                    )
                    ###########################################################
                    print("Detect anomalies", end="\n")

                    score = anomaly.score(observation, metric)
                    score_name = (f"{metric}_percent_{percentage}")

                    if relative is True:
                        score_name = f"{score_name}_relative"

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
                        print("Detect anomalies", end="\n")

                        score = anomaly.score(observation, metric)
                        score_name = (f"{metric}_percent_{percentage}_filter")

                        if relative is True:
                            score_name = f"{score_name}_relative"

                        np.save(f"{save_to}/{score_name}.npy", score)
###############################################################################
# ###############################################################################
# # specobjid to save anomaly scores in data frame
# print("Set meta data tracking")
# train_id_name = parser.get("files", "train_id")
# indexes_interpolate = np.load(f"{input_data_directory}/{train_id_name}")
#
# succesful_interpolation = ~indexes_interpolate[:, 2].astype(bool)
#
# specobjid = indexes_interpolate[succesful_interpolation, 1]
# idx_train_set = indexes_interpolate[succesful_interpolation, 0]
#
# data_frame = pd.DataFrame()
#
# data_frame["specobjid"] = specobjid
# data_frame["trainID"] = idx_train_set
# ###############################################################################
# save_scores = parser.getboolean("parameters", "save_scores")
# output_directory = parser.get("directories", "output")
# output_directory = f"{output_directory}/{metric}"
# check.check_directory(output_directory, exit=False)
#
# for relative in relative_values:
#
#     for percentage in percentage_values:
#
#         print(
#             f"Filter:{filter_lines},Relative:{relative}, {percentage}%",
#             end="\n",
#         )
#
#         anomaly_score = analysis.anomaly_score(
#             metric=metric,
#             observation=observation,
#             percentage=percentage,
#             relative=relative,
#             filter_lines=filter_lines,
#             lines=lines,
#             velocity_filter=velocity_filter,
#             reconstruction_in_drive=reconstruction_in_drive,
#             reconstruction=reconstruction,
#         )
#         #######################################################################
#         # Save anomaly scores
#         score_name = (
#             f"{metric}_relative_{relative}_percentage_{percentage}"
#             f"_filter_{filter_lines}"
#         )
#
#         if filter_lines is True:
#             score_name = f"{score_name}_{velocity_filter}kms"
#
#         save_to = f"{output_directory}/{score_name}"
#
#         # save to data frame
#         data_frame["anomalyScore"] = anomaly_score
#
#         data_frame.to_csv(f"{save_to}.csv.gz", index=False)
#
#         #######################################################################
#         if save_scores:
#
#             np.save(f"{save_to}.npy", anomaly_score)
###############################################################################
# # #######################################################################
# reconstruction_name = parser.get("files", "reconstruction")
# reconstruction_location = f"{model_directory}/{reconstruction_name}"
#
# print("Load reconstructions")
# reconstruction_in_drive = parser.getboolean(
#     "parameters", "reconstruction_in_drive"
# )
#
# if reconstruction_in_drive is False:
#
#     reconstruction = model.reconstruct(observation)
#     np.save(reconstruction_location, reconstruction)
#
#     reconstruction_in_drive = True  # to avoid recomputing it in .mse
#
# else:
#
#     reconstruction = np.load(reconstruction_location)

# # metric parameters
# print("Set parameters of metrics")
# score_items = parser.items("score")
# score_parameters = configuration.section_to_dictionary(score_items, [","])
#
# metric = score_parameters["metric"]
#
# relative_values = score_parameters["relative"]
#
# percentage_values = score_parameters["percentage"]
# #######################################################################


finish_time = time.time()
print(f"Run time: {finish_time - start_time:.2f}")
