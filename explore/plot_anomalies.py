""" Get plots of top 1000 anomalies per score based on residulas"""
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly.constants import GALAXY_LINES
from anomaly.constants import scores_description
from anomaly.plot import inspect_reconstruction
from anomaly.utils import specobjid_to_idx
from autoencoders.ae import AutoEncoder
from sdss.metadata import MetaData

meta = MetaData()
bin_id = "bin_04"
model_id = "0013"
architecture = "256_128_64/latent_12"

meta_data_directory = "/home/edgar/spectra/0_01_z_0_5_4_0_snr_inf"
scores_directory = f"{meta_data_directory}/bin_04/explanation/256_128_64/latent_12"
model_directory = f"{meta_data_directory}/{bin_id}/models/{architecture}"

wave = np.load(f"{meta_data_directory}/wave.npy")
spectra = np.load(f"{meta_data_directory}/spectra.npy", mmap_mode="r")
ids = np.load(f"{meta_data_directory}/ids_inputting.npy")

model = AutoEncoder(
    reload=True,
    reload_from=f"{model_directory}/{model_id}"
)

df_scores = {}

for score_name in scores_description.keys():

    df_scores[score_name] = pd.read_csv(
        f"{scores_directory}/{score_name}/top_normal.csv.gz",
        index_col="specobjid",
    )

image_format = "jpg"

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)

for score in scores_description.keys():


    save_to = f"/home/edgar/anomaly/img/residuals/normal/{score}"

    if os.path.exists(save_to) is False:
        os.makedirs(save_to)

    for rank in range(0, 20):

        print(f"[{score}] Rank: {rank}", end="\r")

        specobjid = df_scores[score].index[rank]
        idx = specobjid_to_idx(specobjid, ids)

        observation = spectra[idx]
        reconstruction = model.reconstruct(observation).reshape(-1)

        inspect_reconstruction(
            wave, observation, reconstruction,
            fig, axs,
            image_format,
            save_to,
            rank
        )
