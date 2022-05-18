"""Data frame and array with top normal and anomalous spectra"""
from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

import numpy as np
import pandas as pd

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "top_scores.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
###############################################################################
data_directory = parser.get("directory", "data")
observation_file = parser.get("file", "observation")
print(f"Load spectra: {observation_file}", end="\n")

observation = np.load(f"{data_directory}/{observation_file}", mmap_mode="r")
###############################################################################
score_directory = parser.get("directory", "score")

metric = parser.get("score", "metric")
filter_velocity = parser.getint("score", "filter")
relative = parser.getboolean("score", "relative")
percentage = parser.get("score", "percentage")

score_name = f"{metric}"

if filter_velocity != 0:

    score_name = f"{score_name}_filter_{filter_velocity}Kms"
    # comply with personal naming convetion of directories
    score_directory = f"{score_directory}/{score_name}"

else:

    # comply with personal naming convetion of directories
    score_directory = f"{score_directory}/{score_name}"

if relative is True:

    score_name = f"{score_name}_rel{percentage}"

else:

    score_name = f"{score_name}_noRel{percentage}"

print(f"Load anomaly scores: {score_name}", end="\n")

score = np.load(f"{score_directory}/{score_name}.npy")

###############################################################################
meta_data_name = parser.get("file", "meta")

print(f"Load data frame: {meta_data_name}", end="\n")

meta_data_directory = parser.get("directory", "meta")

meta_data = pd.read_csv(
    f"{meta_data_directory}/{meta_data_name}", index_col="specobjid"
)
bin_specobjid = score[:, 0].astype(int)
meta_data = meta_data.loc[bin_specobjid]
###############################################################################
save_score_to = parser.get("directory", "output")
save_score_to = f"{save_score_to}/{score_name}"
check.check_directory(save_score_to, exit_program=False)
# colums: ..., the actual score
sort_index = np.argsort(score[:, 2])
bin_specobjid = bin_specobjid[sort_index]

print("Fetch normal spectra",end="\n")
number_normal = parser.getint("spectra", "top_normal")
normal_spectra_index = sort_index[:number_normal]

np.save(
    f"{save_score_to}/top_normal.npy",
    observation[normal_spectra_index],
)
specobjid_normal = bin_specobjid[:number_normal]

normal_meta_data = meta_data.loc[specobjid_normal]
normal_meta_data["score"] = score[sort_index, 2][:number_normal]
normal_meta_data.to_csv(f"{save_score_to}/top_normal.csv.gz")
###############################################################################
print("Fetch anomalies",end="\n")
number_anomalies = parser.getint("spectra", "top_anomalies")
anomalies_spectra_index = sort_index[-1 * number_anomalies:]
np.save(
    f"{save_score_to}/top_anomalies.npy",
    observation[anomalies_spectra_index],
)
specobjid_anomalies = bin_specobjid[-1 * number_anomalies:]

anomalies_meta_data = meta_data.loc[specobjid_anomalies]
anomalies_meta_data["score"] = score[sort_index, 2][-1 * number_anomalies:]
anomalies_meta_data.to_csv(
    f"{save_score_to}/top_anomalies.csv.gz"
)
###############################################################################
# Save configuration file
print(f"Save relevant configuration files", end="\n")
with open(f"{save_score_to}/{config_file_name}", "w") as configfile:
    parser.write(configfile)
###############################################################################
other_config_files = glob.glob(f"{score_directory}/*.ini")

for file_location in other_config_files:

    other_config_file_name = file_location.split("/")[-1]

    with open(file_location, "r") as file:
        other_config_file = file.read()

    with open(f"{save_score_to}/{other_config_file_name}", "w") as file:
        file.write(other_config_file)
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
