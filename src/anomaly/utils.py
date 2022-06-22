"""Functionality such as functions or data containers"""

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
