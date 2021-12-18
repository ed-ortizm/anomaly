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
###############################################################################
class ReconstructionAnomalyScore:
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
    def mse(
        self,
        observation: np.array,
        reconstruction: np.array,
        percentage: int,
        relative: bool,
        epsilon: float,
    ) -> np.array:

        """
        Compute Mean Squared Error between observation and reconstruction.

        PARAMETERS
            observation: array with the origin of fluxes
            reconstruction: the reconstruction of the input observations.
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input
            epsilon: float value to avoid division by zero

        OUTPUT
            anomaly_scor:e of the input observation
        """

        # the square of badly reconstructed spectra mig be larger than
        # a float32
        observation = observation.astype(dtype=float, copy=False)
        reconstruction = reconstruction.astype(dtype=float, copy=False)

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

        flux_wise_error = np.abs(reconstruction - observation)

        if relative:
            flux_wise_error *= 1.0 / (np.abs(reconstruction) + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_mean_value(flux_wise_error, percentage)

        return anomaly_score

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
    def _get_mean_value(
        self, flux_wise_error: np.array, percentage: int
    ) -> np.array:

        """
        Compute mean value of the flux by flux anomaly score

        PARAMETERS
            flux_wise_error: array with reconstruction errors
                flux by flux
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score

        OUTPUT

            mean value of anomaly score of the input observation
        """

        largest_error_ids = self._get_reconstruction_error_ids(
            flux_wise_error, percentage
        )

        anomaly_score = np.empty(largest_error_ids.shape)

        for idx, reconstruction_id in enumerate(largest_error_ids):

            anomaly_score[idx, :] = flux_wise_error[idx, reconstruction_id]

        return np.mean(anomaly_score, axis=1)

    ###########################################################################
    def _get_reconstruction_error_ids(
        self, flux_wise_error: np.array, percentage: int
    ) -> np.array:

        """
        Compute the ids of the pixels with the largest reconstruction
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
    def _reconstruct(self, observation: np.array):

        return self.model.reconstruct(observation)

    ###########################################################################
    def _update_dimensions(self, x: np.array) -> np.array:

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x

    ###########################################################################
