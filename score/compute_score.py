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
import scipy.constants as cst

from anomaly.constants import GALAXY_LINES
from anomaly.metrics import Reconstruction
from autoencoders.ae import AutoEncoder


def compute_mse_scores(spectra: np.ndarray, rec_spectra: np.ndarray):
    """
    Calculates four variants of reconstruction error scores given original
    spectra and their reconstructions.

    Args:
        spectra (np.ndarray): Original input spectra (N_samples, N_features).
        rec_spectra (np.ndarray): Reconstructed spectra from the model
        (N_samples, N_features).

    Returns:
        tuple: A tuple containing 4 numpy arrays (N_samples, 1) in this order:
               (mse_standard, mse_97, mse_relative, mse_97_relative)
    """

    # 1. Standard MSE (Default: 100% of pixels, Absolute)
    mse_standard = Reconstruction().mse(spectra, rec_spectra)

    # 2.Corresponds to: Reconstruction(97, False)
    # This filters out the top 3% most extreme pixel errors before averaging
    mse_97 = Reconstruction(97, False).mse(spectra, rec_spectra)

    # 3. Relative MSE
    # 'True' indicates dividing by the input flux (relative error)
    mse_relative = Reconstruction(100, True).mse(spectra, rec_spectra)

    # 4. Relative MSE 97
    # filters and then computes rel rec error
    mse_97_relative = Reconstruction(97, True).mse(spectra, rec_spectra)

    return mse_standard, mse_97, mse_relative, mse_97_relative

def get_velocity_filter_mask(wave, velocity_filter) -> np.array:

    """
    Compute array with filters for narrow emission lines
    PARAMETERS

        lines: list with lines to discard to compute anomaly_score.
            Check VELOCITY_LINES dictionary at the begin in the document.
        velocity_filter: Doppler velocity to consider at the moment of
            line filtering. It is in units of Km/s.
            DeltaWave = (v/c) * wave

    OUTPUT

        velocity_mask: array of bools with the regions to discard
    """

    c = cst.c * 1e-3  # [km/s]
    alpha = velocity_filter / c  # filter width

    velocity_mask = np.ones(wave.size, dtype=np.bool)

    lines = list(GALAXY_LINES.keys())

    for line in lines:

        delta_wave = GALAXY_LINES[line] * alpha
        # move line to origin
        wave = wave - GALAXY_LINES[line]
        line_mask = (wave < -delta_wave) | (delta_wave < wave)
        # update velocity mask
        velocity_mask *= line_mask

    return velocity_mask

wave = np.load("/home/eortiz/spectra/wave_spectra_imputed.npy")
scores_dir = "/home/eortiz/scores"

print('Load model')

models_dir = "/home/eortiz/models"

ae_model = AutoEncoder(
    reload=True,
    reload_from=f"{models_dir}/bin_03/winner",
)


for bin_id in ['bin_03']:

    spec_bin = np.load(
        f"/home/eortiz/spectra/{bin_id}/{bin_id}_fluxes.npy",
    )

    rec_spec_bin = ae_model.reconstruct(spec_bin)

    np.save(
        f"/home/eortiz/spectra/{bin_id}/{bin_id}_rec_fluxes.npy",
        rec_spec_bin
    )


    print("MSE 4 variations no vel filter")

    (
        mse_standard, mse_97,
        mse_relative, mse_97_relative
    ) = compute_mse_scores(spec_bin, rec_spec_bin)

    np.save(
        f"{scores_dir}/{bin_id}/mse.npy",
        mse_standard
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_97.npy",
        mse_97
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_rel.npy",
        mse_relative
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_97_rel.npy",
        mse_97_relative
    )

    print("MSE 4 variations with vel filter")

    velocity_filter = 250.0
    velocity_mask = get_velocity_filter_mask(wave, velocity_filter)
    spec_bin = spec_bin[:, velocity_mask]
    rec_spec_bin = rec_spec_bin[:, velocity_mask]

    (
        mse_standard, mse_97,
        mse_relative, mse_97_relative
    ) = compute_mse_scores(spec_bin, rec_spec_bin)

    np.save(
        f"{scores_dir}/{bin_id}/mse_filter_{int(velocity_filter)}.npy",
        mse_standard
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_filter_{int(velocity_filter)}_97.npy",
        mse_97
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_filter_{int(velocity_filter)}_rel.npy",
        mse_relative
    )
    np.save(
        f"{scores_dir}/{bin_id}/mse_filter_{int(velocity_filter)}_97_rel.npy",
        mse_97_relative
    )
