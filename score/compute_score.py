"""Compute anomaly scores for thesis"""
import os
# Set environment variables to disable multithreading
# as users will probably want to set the number of cores
# to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

from anomaly.constants import GALAXY_LINES
from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from autoencoders.ae import AutoEncoder


np.seed(0)

bin_id = 'bin_03'

wave = np.load("/home/eortiz/spectra/wave_spectra_imputed.npy")
scores_dir = "/home/eortiz/scores"
# --------------------------------
print('Load model')

models_dir = "/home/eortiz/models"

ae_model = AutoEncoder(
    reload=True,
    reload_from=f"{models_dir}/{bin_id}/winner",
)

spec_bin = np.load(
    f"/home/eortiz/spectra/{bin_id}/{bin_id}_fluxes.npy",
)

rec_spec_bin = ae_model.reconstruct(spec_bin)

np.save(
    f"/home/eortiz/spectra/{bin_id}/{bin_id}_rec_fluxes.npy",
    rec_spec_bin
)
# --------------------------------
print("Compute scores")
lines = list(GALAXY_LINES.keys())
epsilon = 1e-3

velocity_list = [0, 250, 300]
pct_list = 100, 97, 95
isrel_list = [False, True]
# -------------------------------------------------
score_head_name = "mse"

for isrel in isrel_list:

    if isrel is True:
        rel_str = "rel"
    else:
        rel_str = ""

    for velocity_filter in velocity_list:

        if velocity_filter > 0:

            vel_str = f"filter_{int(velocity_filter)}"

        else:
            vel_str = ""

        for pct in pct_list:

            if pct < 100:
                pct_str = f"{pct}"
            else:
                pct_str = ""

            scoring_fn = ReconstructionAnomalyScore(
                reconstruct_function= ae_model.reconstruct,
                filter_parameters=FilterParameters(
                    wave=wave, lines=lines,
                    velocity_filter=velocity_filter
                ),
                reconstruction_parameters=ReconstructionParameters(
                    percentage=pct, relative=isrel, epsilon=epsilon
                ),
            )

            anomaly_score = scoring_fn.score(
                spec_bin, metric='mse'
            )

            score_name = score_head_name
            if vel_str != "":
                score_name += f"_{vel_str}"
            if pct_str != "":
                score_name += f"_{pct_str}"
            if rel_str != "":
                score_name += f"_{rel_str}"

            print(f"Save score: {score_name}")

            np.save(
                f"{scores_dir}/{bin_id}/{score_name}.npy",
                anomaly_score
            )
