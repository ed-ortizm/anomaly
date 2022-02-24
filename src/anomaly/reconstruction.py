import sys

import numpy as np
import scipy.constants as cst
import tensorflow as tf

from .metrics import ReconstructionMetrics
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
class ReconstructionAnomalyScore(ReconstructionMetrics):
    """
    Class to deal with the outliers based on a generative model trained with
    tensorflow.keras
    """

    ###########################################################################
    def __init__(self,
        reconstruct_fucntion,
        lines: list = None,
        wave: np.array,
        velocity_filter: float = None,
        percentage: int = 100,
        relative: bool = False,
        epsilon: float = 1e-3,
    ):
        """
        INPUTS
            reconstruct_fucntion: reconstruct method of trained
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
            epsilon: float value to avoid division by zero
        """

        self.reconstruct = reconstruct_fucntion

        filter_lines = lines != None

        if filter_lines is True:

            self.lines = lines
            self.filter_lines = filter_lines
            self.wave = wave
            self.velocity_filter = velocity_filter

        super().__init__(percentage, relative, epsilon)
    ###########################################################################
    def generalized_lp(
        self,
        observation: np.array,
        p: float=0.33,
    ) -> np.array:

        """
        Compute anomaly score according to metric input parameter

        PARAMETERS
            observation: array with the origin of fluxes
            p: power of lp metric
        OUTPUT
            anomaly_score: of the input observation
        """

        observation, reconstruction = self.reconstruct_and_filter(
            observation,
            self.lines,
            self.velocity_filter
        )

        anomaly_score = super().lp(observation, reconstruction, p)

        return anomaly_score.reshape(-1, 1)
    ###########################################################################
    def maximum_absolute_deviation(
        self,
        observation: np.array,
    ) -> np.array:

        """
        Compute anomaly score according to metric input parameter

        PARAMETERS
            observation: array with the origin of fluxes
        OUTPUT
            anomaly_score: of the input observation
        """

        observation, reconstruction = self.reconstruct_and_filter(
            observation,
            self.lines,
            self.velocity_filter
        )

        anomaly_score = self.super().mad(observation, reconstruction)

        return anomaly_score.reshape(-1, 1)
    ###########################################################################
    def mean_square_error(
        self,
        observation: np.array,
    ) -> np.array:

        """
        Compute anomaly score according to metric input parameter

        PARAMETERS
            observation: array with the origin of fluxes
        OUTPUT
            anomaly_score: of the input observation
        """

        observation, reconstruction = self.reconstruct_and_filter(
            observation,
            self.lines,
            self.velocity_filter
        )

        anomaly_score = super().mse(observation, reconstruction)

        return anomaly_score.reshape(-1, 1)
    ###########################################################################
    def reconstruct_and_filter(self,
        observation:np.array,
        lines: list,
        velocity_filter: float
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

        reconstruction = self.reconstruct(observation)

        if self.filter_lines is True:

            velocity_mask = self.get_velocity_filter_mask(
                lines, velocity_filter
            )

            observation = observation[:, velocity_mask]
            reconstruction = reconstruction[:, velocity_mask]

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

        velocity_mask = self.wave.astype(bool)

        for line in lines:

            delta_wave = GALAXY_LINES[line] * alpha
            # move line to origin
            wave = self.wave - GALAXY_LINES[line]
            line_mask = (wave < -delta_wave) | (delta_wave < wave)
            # update velocity mask
            velocity_mask *= line_mask

        return velocity_mask
