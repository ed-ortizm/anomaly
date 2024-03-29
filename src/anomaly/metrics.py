"""Metrics for outlier detection based on generative models"""
import numpy as np


class Distance:
    """
    Distance metrics between N-dimensioal vectors to compute the
    distance between observation and reconstruction
    """

    def __init__(self, percentage: int):

        self.percentage = percentage

    def correlation(
        self, observation: np.array, reconstruction: np.array
    ) -> np.array:

        """
        Compute correlation distance between observation and reconstruction.
        If u is observation and v reconstruction, then:

        corr = 1
            -
            frac{(u-bar{u})cdot((v-bar{v}))}
            {|(u-bar{u}||(v-bar{v}|}

        PARAMETERS
            observation: array with the origin of fluxes
            reconstruction: the reconstruction of the input observations.
            p:

        OUTPUT
            anomaly_score: of the input observation
        """

        observation = observation.astype(dtype=float)
        observation -= np.mean(observation, axis=1, keepdims=True)

        observation, reconstruction = self._smallest_residuals(
            observation, reconstruction
        )

        reconstruction = reconstruction.astype(dtype=float)
        reconstruction -= np.mean(reconstruction, axis=1, keepdims=True)

        dot_product = np.sum(observation * reconstruction, axis=1)

        observation_norm = np.linalg.norm(observation, axis=1)
        reconstruction_norm = np.linalg.norm(reconstruction, axis=1)

        score = dot_product / (observation_norm * reconstruction_norm)

        score = 1 - score

        return score

    def cosine(
        self, observation: np.array, reconstruction: np.array
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

        observation, reconstruction = self._smallest_residuals(
            observation, reconstruction
        )

        dot_product = np.sum(observation * reconstruction, axis=1)

        observation_norm = np.linalg.norm(observation, axis=1)
        reconstruction_norm = np.linalg.norm(reconstruction, axis=1)

        score = dot_product / (observation_norm * reconstruction_norm)

        score = 1 - score

        return score

    def braycurtis(
        self, observation: np.array, reconstruction: np.array
    ) -> np.array:

        """
        Compute Bray Curtis distance between observation and reconstruction.
        bc: |observation - reconstruction| / |observation + reconstruction|

        PARAMETERS
            observation: array with the origin of fluxes
            reconstruction: the reconstruction of the input observations.
            p:

        OUTPUT
            anomaly_score: of the input observation
        """

        observation = observation.astype(dtype=float)
        reconstruction = reconstruction.astype(dtype=float)

        observation, reconstruction = self._smallest_residuals(
            observation, reconstruction
        )

        flux_diff = np.abs(observation - reconstruction)
        flux_add = np.abs(observation + reconstruction)

        score = np.sum(flux_diff, axis=1)/np.sum(flux_add, axis=1)

        return score.reshape(-1, 1)

    def _smallest_residuals(
        self, observation: np.array, reconstruction: np.array
    ) -> tuple[np.array, np.array]:

        flux_diff = np.abs(observation - reconstruction)
        smallest_error_ids = self._get_smallest_ids(flux_diff)

        for idx, residual_id in enumerate(smallest_error_ids):

            observation[idx, :] = observation[idx, residual_id]
            reconstruction[idx, :] = reconstruction[idx, residual_id]

        return observation, reconstruction

    def _get_smallest_ids(self, flux_diff: np.array) -> np.array:

        """
        Compute the ids of the pixels with the smallest reconstruction
            errors. If percentage is 100, then it does nothing.
            If percentage is 30%, for instance, it returns the ids of
            30% of the pixels with the smalles reconstruction errors.

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

        if self.percentage != 100:

            number_fluxes = int(0.01 * self.percentage * number_fluxes)

            smallest_residuals_ids = np.argpartition(
                flux_diff, number_fluxes, axis=1
            )[:, :number_fluxes]

        else:

            smallest_residuals_ids = np.array(
                [
                    np.arange(0, number_fluxes)
                    for _ in range(flux_diff.shape[0])
                ]
            )

        return smallest_residuals_ids


class Reconstruction:
    """
    Class with metrics to compute anomaly score based on residuals
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
        observation = self._update_dimensions(observation)

        reconstruction = reconstruction.astype(dtype=float)
        reconstruction = self._update_dimensions(reconstruction)

        flux_diff = np.abs(reconstruction - observation)

        flux_diff, observation = self._discard_largest_residuals(
            flux_diff, observation, self.percentage
        )

        flux_diff = flux_diff ** p

        if self.relative is True:
            # chi^2 = (observation - expected value)**2 / expected value
            # In this context each term in the formula means:
            # observation --> reconstruction
            # expected value --> observation
            # notation is a misleading :s
            relative_weight = observation + self.epsilon
            flux_diff *= 1.0 / relative_weight

        anomaly_score = np.sum(flux_diff, axis=1, keepdims=True) ** (1 / p)

        return anomaly_score

    def _discard_largest_residuals(
        self, flux_diff: np.array, observation: np.array, percentage: int
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

        smallest_flux_diff_ids = self._get_smallest_ids(flux_diff, percentage)

        smallest_flux_diff = np.empty(smallest_flux_diff_ids.shape)
        smallest_observation = np.empty(smallest_flux_diff_ids.shape)

        for idx, reconstruction_id in enumerate(smallest_flux_diff_ids):

            smallest_flux_diff[idx, :] = flux_diff[idx, reconstruction_id]

            smallest_observation[idx, :] = observation[idx, reconstruction_id]

        return smallest_flux_diff, smallest_observation

    @staticmethod
    def _get_smallest_ids(flux_diff: np.array, percentage: int) -> np.array:

        """
        Compute the ids of the pixels with the smallest reconstruction
            errors. If percentage is 100, then it does nothing.
            If percentage is 30%, for instance, it returns the ids of
            30% of the pixels with the smalles reconstruction errors.

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

            smallest_residuals_ids = np.argpartition(
                flux_diff, number_fluxes, axis=1
            )[:, :number_fluxes]

        else:

            smallest_residuals_ids = np.array(
                [
                    np.arange(0, number_fluxes)
                    for _ in range(flux_diff.shape[0])
                ]
            )

        return smallest_residuals_ids

    @staticmethod
    def _update_dimensions(x: np.array) -> np.array:

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x
