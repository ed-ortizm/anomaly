import sys

import numpy as np
import scipy.constants as cst
import tensorflow as tf

###############################################################################
GALAXY_LINES = {
    "OII_3727": 3727.0,
    "H_beta_4861": 4861.0,
    "OIII_4959": 4959.0,
    "OIII_5007": 5007.0,
    "NII_6548": 6548.0,
    "H_alpha_6563": 6563.0,
    "NII_6584": 6584,
}
###############################################################################
class ReconstructionAnomalyScore:
    """
    Class for dealing with the outliers based on a generative model trained with
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
    ) -> np.array:

        """
        PARAMETERS

            metric: name of the metric, mse, lp and so on
            observation: array with the origin of fluxes
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input

            lines: list with lines to discard to compute anomaly_score
            filter_lines:
            velocity_filter:

            reconstruction_in_drive:
            reconstruction

            epsilon: float value to avoid division by zero

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

    ###########################################################################
    def get_velocity_filter_mask(
        self, lines: list, velocity_filter: float
    ) -> np.array:

        """
        PARAMETERS
            lines: list with lines to discard to compute anomaly_score
            velocity_filter: wave = (v/c) * wave, ave in amstrong and
                c in km/s

        OUTPUT
            velocity_mask: array of bools with the regions to discard

        """
        c = cst.c * 1e-3  # [km/s]
        z = velocity_filter / c # filter width

        velocity_mask = self.wave.astype(np.bool)

        for line in lines:

            delta_wave = GALAXY_LINES[line] * z
            # move line to origin
            wave = self.wave - GALAXY_LINES[line]
            line_mask = (wave < -delta_wave) * (delta_wave < wave)
            # update velocity mask
            velocity_mask *= line_mask

        return velocity_mask

    ###########################################################################
    def mse(
        self,
        observation: np.array,
        reconstruction: np.array,
        percentage: int,
        relative: bool,
        epsilon,
    ) -> np.array:

        """
        PARAMETERS
            observation: array with the origin of fluxes
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input
            epsilon: float value to avoid division by zero
        OUTPUT
            anomaly score of the input observation
        """

        flux_wise_error = (reconstruction - observation) ** 2.0

        if relative is True:
            flux_wise_error *= 1.0 / (reconstruction ** 2.0 + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_mean_value(flux_wise_error, percentage)

        return anomaly_score

    ###########################################################################
    def mad(
        self,
        observation: np.array,
        reconstruction: np.array,
        percentage: int,
        relative: bool,
        epsilon: float,
    ) -> np.array:

        """
        PARAMETERS
            observation: array with the origin of fluxes
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input
            epsilon: float value to avoid division by zero
        OUTPUT
            anomaly score of the input observation
        """

        reconstruction = self._reconstruct(observation)
        flux_wise_error = np.abs(reconstruction - observation)

        if relative:
            flux_wise_error *= 1.0 / (np.abs(reconstruction) + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_mean_value(flux_wise_error, percentage)

        return anomaly_score

    ###########################################################################
    def _reconstruct(self, observation: np.array):

        return self.model.reconstruct(observation)

    ###########################################################################
    def _update_dimensions(self, x: "np.array") -> np.array:

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x

    ###########################################################################
    def _get_reconstruction_error_ids(
        self, flux_wise_error: np.array, percentage: int
    ) -> np.array:

        """
        Computes the ids of the pixels with the largest reconstruction
            errors. If percentage is 100, then it does nothing.
            If percentage is 30%, for instance, it returns the ids of
            30% of the pixels with the highest reconstruction errors.

        PARAMETERS
            flux_wise_error: array with reconstruction errors
                flux by flux
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
        OUTPUT
            largest_reconstruction_error_ids: ids with the percentage
                of pixels with the highest reconstruction errors
        """

        number_fluxes = flux_wise_error.shape[1]
        number_anomalous_fluxes = int(0.01 * percentage * number_fluxes)

        largest_reconstruction_error_ids = np.argpartition(
            flux_wise_error, -number_anomalous_fluxes, axis=1
        )[:, -number_anomalous_fluxes:]

        return largest_reconstruction_error_ids

    ###########################################################################
    def _get_mean_value(
        self, flux_wise_error: np.array, percentage: int
    ) -> np.array:

        """
        PARAMETERS
            flux_wise_error: array with reconstruction errors
                flux by flux
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
        OUTPUT
            mean anomaly score of the input observation
        """

        reconstruction_error_ids = self._get_reconstruction_error_ids(
            flux_wise_error, percentage
        )

        anomaly_score = np.empty(reconstruction_error_ids.shape)

        for idx, reconstruction_id in enumerate(reconstruction_error_ids):

            anomaly_score[idx, :] = flux_wise_error[idx, reconstruction_id]

        return np.mean(anomaly_score, axis=1)

    ###########################################################################
