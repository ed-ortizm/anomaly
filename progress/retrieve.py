import glob

import numpy as np
import pandas as pd

from sdss.superclasses import FileDirectory
###############################################################################
def save_directory(name_frame, analysis_directory):

    if "filter_True" in name_frame:

        if "relative_True" in name_frame:

            save_to = f"{analysis_directory}/filter/relative"

        else:

            save_to = f"{analysis_directory}/filter/norelative"

    else:

        if "relative_True" in name_frame:

            save_to = f"{analysis_directory}/nofilter/relative"

        else:

            save_to = f"{analysis_directory}/nofilter/norelative"

    return save_to
###############################################################################
check = FileDirectory()
###############################################################################
# Constant
work_directory = "/home/edgar/spectra/process"
anomaly_directory = "anomaly/mse"
###############################################################################
meta_data = "0_01_z_inf_10_0_snr_20_0"
model = "vae"
architecture = "1000_500_100_6_100_500_1000"
model = f"{model}/{architecture}"
###############################################################################
# data
observation = np.load(f"{work_directory}/{meta_data}/fluxes.npy")

reconstruction = np.load(
    f"{work_directory}/{meta_data}/{model}/reconstructions.npy"
)
###############################################################################

frames = glob.glob(
    f"{work_directory}/{meta_data}/{model}/{anomaly_directory}/*percentage_100*.gz"
)
object_types = ["normal", "middle", "anomaly"]
###############################################################################
analysis_directory = f"/home/edgar/anomaly/progress/analysis"
check.check_directory(analysis_directory, exit=False)

for frame_location in frames:


    data = pd.read_csv(frame_location, index_col="specobjid")
    data.sort_values(by=["anomalyScore"], inplace=True)

    name_frame = frame_location.split("/")[-1]
    save_to = save_directory(name_frame, analysis_directory)
    check.check_directory(save_to, exit=False)
    check.check_directory(f"{save_to}/score", exit=False)

    for object_type in object_types:

        if object_type == "normal":

            score = data["anomalyScore"].values[:10_000]

            train_id = data["trainID"].values[:10_000]
            obs = observation[train_id]
            rec = reconstruction[train_id]

        elif object_type == "anomaly":

            score = data["anomalyScore"].values[-10_000:]

            train_id = data["trainID"].values[-10_000:]
            obs = observation[train_id]
            rec = reconstruction[train_id]

        elif object_type == "middle":

            score = data["anomalyScore"].values[195_000:205_000]

            train_id = data["trainID"].values[195_000:205_000]
            obs = observation[train_id]
            rec = reconstruction[train_id]

        check.check_directory(f"{save_to}/{object_type}", exit=False)

        np.save(
            f"{save_to}/{object_type}/{object_type}_observation_{name_frame[:-7]}.npy", obs
        )
        np.save(
            f"{save_to}/{object_type}/{object_type}_reconstruction_{name_frame[:-7]}.npy", rec)

        np.save(
            f"{save_to}/score/{object_type}_score_{name_frame[:-7]}.npy", score
        )
