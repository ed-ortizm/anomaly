"""Functionality and data containers"""

from collections import namedtuple

import numpy as np
import scipy.constants as cst
# pylint: disable=E0611
from skimage.color import gray2rgb

from anomaly.constants import GALAXY_LINES


FilterParameters = namedtuple(
    "FilterParameters", ["wave", "velocity_filter", "lines"]
)

ReconstructionParameters = namedtuple(
    "ReconstructionParameters", ["relative", "percentage", "epsilon"]
)


def spectra_to_batch_image(spectra):
    """
    Convert spectra to a batch of RGB images where the height
    of an spectrum's image is 1. The output shae will be:
    (batch_id, 1, flux, 3)

    """

    # If a 1D spec is passed
    if spectra.ndim == 1:
        # get (1, flux)
        gray_spectra = spectra[np.newaxis, ...]
        # get (1, flux, 3)
        spectra_image = gray2rgb(gray_spectra)
        # get (n_batch, 1, flux, 3)
        return spectra_image[np.newaxis, ...]
    # array of spectra: (n_batch, flux)
    if spectra.ndim == 2:
        # get (n_bacth, flux, 3)
        gray_spectra = gray2rgb(spectra)
        # get (n_bacth, 1, flux, 3)
        return gray_spectra[:, np.newaxis, ...]
    # if already image pass to (n_batch, 1, flux, 3)
    if spectra.ndim == 3:
        return spectra[np.newaxis, ...]

    return spectra

class VelocityFilter:
    """
    Handle filter operations according to provided lines and
    velocity width
    """

    def __init__(
        self,
        wave: np.array,
        velocity_filter: float = 0.0,
        lines: list = None,
    ):

        self.wave = wave
        self.lines = lines
        self.velocity_filter = velocity_filter

    def filter(self, spectra: np.array) -> tuple:

        """
        PARAMETERS
            observation: array with the origin of fluxes
            lines: list with lines to discard to compute anomaly_score
            velocity_filter: Doppler velocity to consider at the moment of
                line filtering. It is in units of Km/s.
                DeltaWave = (v/c) * wave

        OUTPUTS
            observation, reconstruction:
                np.arrays with the filter if it applies
        """

        velocity_mask = self.get_velocity_filter_mask()

        spectra = spectra[:, velocity_mask]

        return spectra

    def get_velocity_filter_mask(self) -> np.array:

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
        alpha = self.velocity_filter / c  # filter width

        velocity_mask = np.ones(self.wave.size, dtype=bool)

        for line in self.lines:

            delta_wave = GALAXY_LINES[line] * alpha
            # move line to origin
            wave = self.wave - GALAXY_LINES[line]
            line_mask = (wave < -delta_wave) | (delta_wave < wave)
            # update velocity mask
            velocity_mask *= line_mask

        return velocity_mask

def specobjid_to_idx(specobjid: int, ids: np.array) -> int:
    """
    Obtain index of spectrum in array that contains all
    spectra (non-binned) already preprocessed.

    INPUTS
    specobjid: unique sdss id indentifier
    ids: array that relates specobjid with the index of the
        spectrum in array with all spectra.
        ids[:, 0] -> index in spectra array
        ids[:, 1] -> specobjid of spectra

    OUTPUT
    idx: index of specobjid spectrum in array with all spectra

    """

    mask = np.where(ids[:, 1] == specobjid, True, False)

    idx = int(ids[mask, 0][0])

    return idx

def line_width_from_velocity(velocity: float, line_wavelength: float) -> float:

    """
    Get the width of a line in Angstrom accoring to the input
    rotational velocity.

    INPUT
    velocity: rotational velocity in kms^-1
    line_wavelength: wavelength of emissionon line to test,
        e.g $H_{\alpha}$ = 6562

    OUTPUT
    line_width: width of line in Angstrom
    """

    # convert light speed to kms-1
    c = cst.c * 1e-3

    line_width = 2 * (velocity / c) * line_wavelength

    return line_width

class AnomalyOverlapAnalyzer:
    """
    A utility class for performing set operations (intersections, differences)
    on anomaly detection scores.
    """

    def __init__(self):
        # As requested, no input arguments are needed for initialization
        pass

    @staticmethod
    def overlap_pair_scores(score_a, score_b, df, quantile=99):
        """
        Computes the overlap and differences between the top percentile 
        of two specific scores.
        """
        # Convert integer quantile (e.g., 99) to float (0.99)
        q_val = quantile * 0.01

        thresh_a = df[score_a].quantile(q_val)
        thresh_b = df[score_b].quantile(q_val)

        ids_a = set(df[df[score_a] > thresh_a].index)
        ids_b = set(df[df[score_b] > thresh_b].index)

        # 1. Intersection (objects present in BOTH sets)
        common_ids = ids_a.intersection(ids_b)

        # 2. Non-Common (Unique to each score)
        only_in_a = ids_a - ids_b  # Present in A, but NOT in B
        only_in_b = ids_b - ids_a  # Present in B, but NOT in A

        return ids_a, ids_b, common_ids, only_in_a, only_in_b

    @staticmethod
    def get_unique_ids(ids_dict, score_list):
        """
        Identifies IDs unique to each score within a specific group of scores.
        (e.g., found by 'mse' but NOT by any other score in the list).
        """
        unique_ids_dict = {}

        for target in score_list:
            # Collect all sets EXCEPT the current target
            other_sets = [ids_dict[k] for k in score_list if k != target]

            # Subtract all other sets from the target set
            unique_ids_dict[target] = ids_dict[target].difference(*other_sets)

        return unique_ids_dict

    @staticmethod
    def get_core_common_ids(ids_dict, score_list):
        """
        Identifies IDs that are present in ALL sets for the provided
        list of scores.
        (The intersection of the entire group).
        """
        # Collect all sets corresponding to the score list
        all_sets = [ids_dict[k] for k in score_list]

        # Compute the intersection of the entire group
        core_common_ids = set.intersection(*all_sets)

        return core_common_ids
