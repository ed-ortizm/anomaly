import sys

import numpy as np
import scipy.constants as cst
import tensorflow as tf

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
##############################################################################
class ReconstructionAnomalyScore(Metrics):
    """
    Class to deal with the outliers based on a generative model trained with
    tensorflow.keras
    """

    ###########################################################################
    def __init__(self, model: tf.keras.Model, wave: np.array):
        """
        INPUTS
            model: trained generative model with reconstruct method
            wave: common grid to spectra
        """

        self.model = model
        self.wave = wave

    ###########################################################################
    def anomaly_score(
        self,
        metric: str,
        observation: np.array,
        percentage: int,
        relative: bool,
        filter_lines: bool,
        lines: list = None,
        velocity_filter: float = None,
        reconstruction_in_drive: bool = False,
        reconstruction: np.array = None,
        epsilon: float = 1e-3,
        p: float = None,
    ) -> np.array:

        """
        Compute anomaly score according to metric input parameter

        PARAMETERS

            metric: name of the metric, mse, lp and so on
            observation: array with the origin of fluxes

            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score

            relative: whether or not the score is weigthed by the input

            lines: list with lines to discard to compute anomaly_score

            filter_lines: True indicates the score is computed with
                lines filtered

            velocity_filter: Doppler velocity to consider at the moment of
                line filtering. It is in units of Km/s.
                DeltaWave = (v/c) * wave

            reconstruction_in_drive: if True, there is no need to generate the
                reconstruction of imput observation.
                Comes in handy to analyse large array of observations.

            reconstruction: the reconstruction of the input observations.

            epsilon: float value to avoid division by zero

            p: value for lp metric

        OUTPUT
            anomaly_score: of the input observation
        """

        if reconstruction_in_drive is False:
            reconstruction = self._reconstruct(observation)
        #######################################################################
        if filter_lines is True:

            velocity_mask = self.get_velocity_filter_mask(
                lines, velocity_filter
            )

            observation = observation[:, velocity_mask]
            reconstruction = reconstruction[:, velocity_mask]
        #######################################################################
        if metric == "mse":

            anomaly_score = self.mse(
                observation, reconstruction, percentage, relative, epsilon
            )

            return anomaly_score
        #######################################################################
        if metric == "mad":

            anomaly_score = self.mad(
                observation, reconstruction, percentage, relative, epsilon
            )

            return anomaly_score
        #######################################################################
        if metric == "lp":

            assert isscalar(p)

            anomaly_score = self.lp(
                observation, reconstruction, p, percentage, relative, epsilon
            )

            return anomaly_score

    ###########################################################################

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

        velocity_mask = self.wave.astype(bool)

        for line in lines:

            delta_wave = GALAXY_LINES[line] * alpha
            # move line to origin
            wave = self.wave - GALAXY_LINES[line]
            line_mask = (wave < -delta_wave) | (delta_wave < wave)
            # update velocity mask
            velocity_mask *= line_mask

        return velocity_mask

    ###########################################################################

    ###########################################################################
    def _reconstruct(self, observation: np.array) -> np.array:

        return self.model.reconstruct(observation)
