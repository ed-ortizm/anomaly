"""Top normal and anomalous spectra"""
from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

import numpy as np

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "data_to_explain.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
###############################################################################
# Load data
score_directory = parser.get("directory", "score")

metric = parser.get("score", "metric")
filter_velocity = parser.getint("score", "filter")
relative = parser.getboolean("score", "relative")
percentage = parser.get("score", "percentage")

score_name = f"{metric}"
if filter_velocity != 0:
    score_name = f"{score_name}_filter_{filter_velocity}Kms_"
if relative is True:
    score_name = f"{score_name}_rel{percentage}"
else:
    score_name = f"{score_name}_noRel{percentage}"

score_directory = f"{score_directory}/{metric}"
score = np.load(f"{score_directory}/{score_name}.npy")

print(f"Load anomaly scores: {score_name}", end="\n")

# colums: ..., the actual score
sort_index = np.argsort(score[:, 2])
# score = score[sort_index]

save_from_index = int(1000)

save_score_to = parser.get("directory", "output")
# check.check_directory(save_to, exit_program=False)
save_score_to = f"{save_score_to}/{score_name}"
check.check_directory(save_score_to, exit_program=False)

np.save(
    f"{save_score_to}/{score_name}.npy",
    score[sort_index][-save_from_index:],
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
