"""
Module with functionality to compute anomaly scores based on reconstructions
"""
from colletions import namedtuple
import sys

import numpy as np
import scipy.constants as cst
from skimage.color import gray2rgb  # convert spectra to 3 channels

from anomaly.metrics import Distance, Reconstruction

###############################################################################
GALAXY_LINES = {
    # EMISSION
    "OII_3727": 3727.0,
    "H_delta_4102": 4102.0,
    "H_gamma_4340": 4340.0,
    "H_beta_4861": 4861.0,
    "OIII_4959": 4959.0,
    "OIII_5007": 5007.0,
    "NII_6548": 6548.0,
    "H_alpha_6563": 6563.0,
    "NII_6584": 6584.0,
    "SII_6716": 6716.0,
    "SII_6720": 6720.0,
    "SII_6731": 6731.0,
    # ABSORPTION
}

class VelocityFilter:

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

class DistanceAnomalyScore(Distance):
    """
    Class to deal with the outliers based on a generative model trained with
    tensorflow.keras and distance metrics different from metrics based on
    reconstruction residuals
    """
    def __init__(self,
        reconstruct_function,
        filter_parameters: namedtuple,
    ):

        """
        INPUTS
            reconstruct_function: reconstruct method of trained
                generative model

            filter_parameters: named tuple containing

                lines: list with lines to discard to compute anomaly_score
                velocity_filter: Doppler velocity to consider at the moment of
                    line filtering. It is in units of Km/s.
                    DeltaWave = (v/c) * wave
                wave: common grid to spectra
        """

        self.reconstruct = reconstruct_function

        velocity_filter = filter_parameters.velocity_filter

        self.filter_object = VelocityFilter(
            wave=filter_parameters.wave,
            velocity_filter=velocity_filter,
            lines=filter_parameters.lines
        )

        self.filter_lines = velocity_filter != 0

        super().__init__()

    def score(
        self, observation: np.array, metric: str
    ) -> np.array:
        """Compute reconstruction error """

        # in case I pass a spectra with one dimension
        # this line converts 1D array to (1, n_wave, 3)
        # an image where each channel has the spectrun
        observation = self.spectra_to_batch_image(observation)

        assert observation.ndim == 4

        reconstruction = self.reconstruct(observation[:, 0, :, 0])

        if self.filter_lines is True:

            observation = self.filter_object.filter(observation[:, 0, :, 0])
            reconstruction = self.filter_object.filter(reconstruction)

        if metric == "correlation":

            anomaly_score = super().correlation(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        if metric == "cosine":

            anomaly_score = super().cosine(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        print(f"{metric} not implemented")
        sys.exit()

    ###########################################################################
    @staticmethod
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
            return spectra_image[np.newaxis]
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


class ReconstructionAnomalyScore(Reconstruction):
    """
    Class to deal with the outliers based on a generative model
    trained with tensorflow.keras. The class uses metrics based
    on reconstruction residuals
    """

    ###########################################################################
    def __init__(
        self,
        reconstruct_function,
        wave: np.array,
        lines: list = None,
        velocity_filter: float = 0.,
        percentage: int = 100,
        relative: bool = False,
    ):
        """
        INPUTS
            reconstruct_function: reconstruct method of trained
                generative model

            lines: list with lines to discard to compute anomaly_score
            velocity_filter: Doppler velocity to consider at the moment of
                line filtering. It is in units of Km/s.
                DeltaWave = (v/c) * wave
            wave: common grid to spectra

            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input
        """

        self.reconstruct = reconstruct_function
        self.wave = wave

        self.lines = lines
        self.filter_lines = velocity_filter != 0
        self.velocity_filter = velocity_filter

        super().__init__(percentage=percentage, relative=relative)

    ###########################################################################
    def score(
        self, observation: np.array, metric: str, p: float = 0.33
    ) -> np.array:
        """Compute reconstruction error """

        # in case I pass a spectra with one dimension
        # this line converts 1D array to (1, n_wave, 3)
        # an image where each channel has the spectrun
        observation = self.spectra_to_batch_image(observation)

        assert observation.ndim == 4

        observation, reconstruction = self.reconstruct_and_filter(
            observation, self.lines, self.velocity_filter
        )

        # make compatible batch of spectra's images with metrics
        observation = observation[:, 0, :, 0]
        reconstruction = reconstruction[:, 0, :, 0]
        if metric == "lp":

            assert np.isscalar(p)

            anomaly_score = super().lp(observation, reconstruction, p)
            return anomaly_score.reshape((-1, 1))

        if metric == "mse":

            anomaly_score = super().mse(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        if metric == "mad":

            anomaly_score = super().mad(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        print(f"{metric} not implemented")
        sys.exit()

    ###########################################################################
    def reconstruct_and_filter(
        self, observation: np.array, lines: list, velocity_filter: float
    ) -> tuple:

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

        # get (n_batch, flux). This is what is compatible with
        # reconstruction method
        reconstruction = self.reconstruct(observation[:, 0, :, 0])

        if self.filter_lines is True:

            velocity_mask = self.get_velocity_filter_mask(
                lines, velocity_filter
            )

            observation = observation[:, 0, velocity_mask, 0]
            reconstruction = reconstruction[:, velocity_mask]

        observation = self.spectra_to_batch_image(observation)
        reconstruction = self.spectra_to_batch_image(reconstruction)

        assert observation.ndim == 4
        assert reconstruction.ndim == 4

        return observation, reconstruction

    ###########################################################################
    def get_velocity_filter_mask(
        self, lines: list, velocity_filter: float
    ) -> np.array:

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

        velocity_mask = np.ones(self.wave.size, dtype=np.bool)

        for line in lines:

            delta_wave = GALAXY_LINES[line] * alpha
            # move line to origin
            wave = self.wave - GALAXY_LINES[line]
            line_mask = (wave < -delta_wave) | (delta_wave < wave)
            # update velocity mask
            velocity_mask *= line_mask

        return velocity_mask

    ###########################################################################
    @staticmethod
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
            return spectra_image[np.newaxis]
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
