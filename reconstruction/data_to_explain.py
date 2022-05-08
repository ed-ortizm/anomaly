from configparser import ConfigParser, ExtendedInterpolation
import glob
import shutil
import time

import numpy as np

from sdss.superclasses import FileDirectory, ConfigurationFile

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
data_directory = parser.get("directory", "data")
files_location = glob.glob(f"{data_directory}/[a-z]*/*.npy")

print(f"Load anomaly scores", end="\n")

percentage = parser.getint("configuration", "percentage")
percentage /= 100

save_to = parser.get("directory", "save_to")
check.check_directory(save_to, exit=False)

for file in files_location:

    name_file = file.split("/")[-1].split(".")[0]

    print(f"Load {name_file}", end="\r")

    score = np.load(file)

    # colums: ..., the actual score
    sort_index = np.argsort(score[:, 2])
    # score = score[sort_index]

    save_from_index = int(percentage * score.shape[0])

    save_score_to = f"{save_to}/{name_file}"
    check.check_directory(save_score_to, exit=False)

    np.save(
        f"{save_score_to}/{name_file}.npy",
        score[sort_index][-save_from_index:],
    )

###############################################################################
# Save configuration file
print(f"Save relevant configuration files", end="\n")
with open(f"{save_to}/{config_file_name}", "w") as configfile:
    parser.write(configfile)
###############################################################################
other_config_files = glob.glob(f"{data_directory}/*.ini")
for file in other_config_files:
    file_name = file.split("/")[-1]
    shutil.copyfile(file, f"{save_to}/{file_name}")
###############################################################################
finish_time = time.time()
print(f"\nRun time: {finish_time - start_time:.2f}")
