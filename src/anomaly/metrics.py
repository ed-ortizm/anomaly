"""Metrics for outlier detection based on generative models"""
import numpy as np

###############################################################################
class ReconstructionMetrics:
    """
    Class with metrics to compute anomaly score based on a reconstruction
    """

    def __init__(
        self,
        percentage: int = 100,
        relative: bool = False,
        epsilon: float = 1e-3,
    ):
        """
        PARAMETERS
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
            relative: whether or not the score is weigthed by the input
            epsilon: float value to avoid division by zero
        """

        self.percentage = percentage
        self.relative = relative
        self.epsilon = epsilon

    @staticmethod
    def cosine(
        observation: np.array, reconstruction: np.array
    ) -> np.array:

        """
        Compute cosine distance between observation and reconstruction

        PARAMETERS
            observation: array with the origin of fluxes
            reconstruction: the reconstruction of the input observations.
            p:

        OUTPUT
            anomaly_score: of the input observation
        """

        observation = observation.astype(dtype=float)
        reconstruction = reconstruction.astype(dtype=float)

        dot_product = np.sum(observation*reconstruction, axis=1)

        observation_norm = np.linalg.norm(observation, axis=1)
        reconstruction_norm = np.linalg.norm(reconstruction, axis=1)

        anomaly_score = dot_product/(observation_norm*reconstruction_norm)

        anomaly_score = 1 - anomaly_score

        return anomaly_score

    def mse(self, observation: np.array, reconstruction: np.array) -> np.array:

        """
        Compute Mean Squared Error between observation and reconstruction.

        PARAMETERS
            observation: array with the original of fluxes
            reconstruction: the reconstruction of the input observations.

        OUTPUT
            anomaly_score of the input observation
        """

        return self.lp(observation, reconstruction, p=2)

    def mad(self, observation: np.array, reconstruction: np.array) -> np.array:

        """
        PARAMETERS
            observation: array with the original of fluxes
            reconstruction: the reconstruction of the input observations.

        OUTPUT
            anomaly_score of the input observation
        """

        return self.lp(observation, reconstruction, p=1)

    def lp(
        self, observation: np.array, reconstruction: np.array, p: float = 0.33
    ) -> np.array:

        """
        Compute LP between observation and reconstruction.

        PARAMETERS
            observation: array with the origin of fluxes
            reconstruction: the reconstruction of the input observations.
            p:

        OUTPUT
            anomaly_score: of the input observation
        """

        # the square of badly reconstructed spectra mig be larger than
        # a float32
        observation = observation.astype(dtype=float)
        reconstruction = reconstruction.astype(dtype=float)

        flux_diff = np.abs(reconstruction - observation) ** p

        if self.relative is True:

            relative_weight = np.abs(reconstruction) ** (1 / p) + self.epsilon
            flux_diff *= 1.0 / relative_weight

        flux_diff = self._update_dimensions(flux_diff)

        anomaly_score = self._get_mean_value(flux_diff, self.percentage)

        return anomaly_score

    ###########################################################################
    def _get_mean_value(
        self, flux_diff: np.array, percentage: int
    ) -> np.array:

        """
        Compute mean value of the flux by flux anomaly score

        PARAMETERS
            flux_diff: array with reconstruction errors
                flux by flux
            percentage: of fluxes with the highest contribution to the
                anomaly score

        OUTPUT

            mean value of anomaly score of the input observation
        """

        smallest_error_ids = self._get_reconstruction_error_ids(
            flux_diff, percentage
        )

        anomaly_score = np.empty(smallest_error_ids.shape)

        for idx, reconstruction_id in enumerate(smallest_error_ids):

            anomaly_score[idx, :] = flux_diff[idx, reconstruction_id]

        return np.mean(anomaly_score, axis=1)

    ###########################################################################
    @staticmethod
    def _get_reconstruction_error_ids(
        flux_diff: np.array, percentage: int
    ) -> np.array:

        """
        Compute the ids of the pixels with the largest reconstruction
            errors. If percentage is 100, then it does nothing.
            If percentage is 30%, for instance, it returns the ids of
            30% of the pixels with the highest reconstruction errors.

        PARAMETERS
            flux_diff: array with reconstruction errors
                flux by flux
            percentage: of fluxes with the highest contribution to the
                anomaly score
        OUTPUT
            largest_reconstruction_error_ids: ids with the percentage
                of pixels with the highest reconstruction errors
        """

        number_fluxes = flux_diff.shape[1]

        if percentage != 100:

            number_fluxes = int(0.01 * percentage * number_fluxes)

        else:

            number_fluxes -= 1

        smallest_residuals_ids = np.argpartition(
            flux_diff, number_fluxes, axis=1
        )[:, :number_fluxes]


        return smallest_residuals_ids

    ###########################################################################
    @staticmethod
    def _update_dimensions(x: np.array) -> np.array:

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x

    ###########################################################################
