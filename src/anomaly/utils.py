"""Functionality and data containers"""

from collections import namedtuple

import numpy as np
import scipy.constants as cst
from skimage.color import gray2rgb  # convert spectra to 3 channels

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

    def __init__(self,
        wave: np.array,
        velocity_filter: float = 0.,
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

        velocity_mask = np.ones(self.wave.size, dtype=np.bool)

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

    mask = np.where(ids[:, 1]==specobjid, True, False)
    
    idx = int(ids[mask, 0][0])

    return idx

def set_intersection(specobjids: dict, max_rank: int, min_rank: int) -> set:

    """
    Interset sets of specobjids
    
    INPUT
    specobjids: dictionary with arrays of specobjids ordered by anomalous
        rank
        key: score name, e.g, lp_noRel100
        value: set with specobjid of spectra
    max_rank: the (max_rank+1)^{th} most anomalous spectrum
    min_rank: the (min_rank+1)^{th} most anomalous spectrum
        If the rank is 0 then it is the most anomalous spectrum
        If the rank is 1,then it is the second mosth anomalous spectrum...
    """
    score_names = list(specobjids.keys())
    # Adjust ranks
    if min_rank == 0:

        specobjids = {
            score_name: specobjids[score_name][-max_rank:] for score_name in score_names
            }

    else:

        specobjids = {
            score_name: specobjids[score_name][-max_rank:-min_rank] for score_name in score_names
            }
    
    # get a set of any score
    intersection_set = set(specobjids[score_names[0]])

    for score_name in score_names:

        intersection_set = intersection_set.intersection(
            set(specobjids[score_name])
        )
    
    return intersection_set

def set_difference(specobjids: set, intersection_specobjids: set) -> set:
    """
    Return the set diffrerence between specobjids and a set that
    came from the intersection with other sets
    
    INPUTS
    specobjids: set with ids that was used for an intersection
    intersection_specobjids: set from the intersection of
        specobjids with other sets
    """

    set_difference = specobjids.difference(intersection_specobjids)
    
    return set_difference