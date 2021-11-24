import sys
import numpy as np

###############################################################################
class ReconstructionAnomalyScore:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    ###########################################################################
    def __init__(self, model: "tf.keras.model"):
        """
        INPUTS
            model: trained generative model with reconstruct method
        OUTPUT
            object
        """
        self.model = model

    ###########################################################################
    def anomaly_score(
        self,
        metric: str,
        observation: np.array,
        percentage: int,
        relative: bool,
        filter_narrow_lines:bool,
        velocity_filter: float=None,
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
            epsilon: float value to avoid division by zero
        OUTPUT
            anomaly_score: of the input observation
        """

        if filter_narrow_lines:

            observation = self.filter_narrow_lines(
                observation,
                velocity_filter,
            )

            ###################################################################
            if reconstruction_in_drive:

                reconstruction = self.filter_narrow_lines(
                    reconstruction,
                    velocity_filter,
                )

        #######################################################################
        if metric == "mse":

            anomaly_score = self.mse(
                observation=observation,
                percentage=percentage,
                relative=relative,
                reconstruction_in_drive=reconstruction_in_drive,
                reconstruction=reconstruction,
            )

            return anomaly_score
        #######################################################################
        if metric == "mad":

            anomaly_score = self.mad(
                observation=observation,
                percentage=percentage,
                relative=relative,
                reconstruction_in_drive=reconstruction_in_drive,
                reconstruction=reconstruction,
            )

            return anomaly_score
        #######################################################################

    ###########################################################################
    def filter_narrow_lines(self,
        observation:np.array,
        velocity_filter,
    )-> np.array:
        pass
    ###########################################################################
    def mse(
        self,
        observation: np.array,
        percentage: int,
        relative: bool,
        reconstruction_in_drive: bool,
        reconstruction: np.array,
        epsilon: float = 1e-3,
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

        if not reconstruction_in_drive:
            reconstruction = self._reconstruct(observation)

        flux_wise_error = np.square(reconstruction - observation)

        if relative:
            flux_wise_error *= 1.0 / np.square(reconstruction + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_anomaly_score(flux_wise_error, percentage)

        return anomaly_score

    ###########################################################################
    def mad(
        self,
        observation: np.array,
        percentage: int,
        relative: bool = False,
        epsilon: float = 1e-3,
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
            flux_wise_error *= 1.0 / np.abs(reconstruction + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_anomaly_score(flux_wise_error, percentage)

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
            flux_wise_error, -1 * number_anomalous_fluxes, axis=1
        )[:, -1 * number_anomalous_fluxes :]

        return largest_reconstruction_error_ids

    ###########################################################################
    def _get_anomaly_score(
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


#     def _lp(self, O, R, percentage, image):
#
#         if image:
#             O, R = O[:, 0, :, 0], R[:, 0, :, 0]
#
#         reconstruction = np.abs(R - O)**self.p
#
#         number_outlier_fluxes = int(0.01*percentage*reconstruction.shape[1])
#
#         largest_reconstruction_ids = np.argpartition(
#             reconstruction,
#             -1 * number_outlier_fluxes,
#             axis=1)[:, -1 * number_outlier_fluxes:]
#
#         score = np.empty(largest_reconstruction_ids.shape)
#
#         for n, idx in enumerate(largest_reconstruction_ids):
#
#             score[n, :] = reconstruction[n, idx]
#
#         o_score = np.sum(score, axis=1)**(1 / self.p)
#
#         if image:
#
#             similarity =  o_score.max() - o_score
#
#             o_similarity = np.empty((o_score.size, 2))
#             o_similarity[:, 0] = o_score[:]
#             o_similarity[:, 1] = similarity[:]
#
#             return o_similarity
#
#         return o_score
#     ############################################################################
#     def _lp_relative(self, O, R, percentage, image):
#
#         if image:
#             O, R = O[:, 0, :, 0], R[:, 0, :, 0]
#
#         reconstruction = np.abs( (R - O)/(O + self.epsilon) )**self.p
#
#         number_outlier_fluxes = int(0.01*percentage*reconstruction.shape[1])
#
#         largest_reconstruction_ids = np.argpartition(
#             reconstruction,
#             -1 * number_outlier_fluxes,
#             axis=1)[:, -1 * number_outlier_fluxes:]
#
#         score = np.empty(largest_reconstruction_ids.shape)
#
#         for n, idx in enumerate(largest_reconstruction_ids):
#
#             score[n, :] = reconstruction[n, idx]
#
#         o_score = np.sum(score, axis=1)**(1 / self.p)
#
#         if image:
#
#             similarity =  o_score.max() - o_score
#
#             o_similarity = np.empty((o_score.size, 2))
#             o_similarity[:, 0] = o_score[:]
#             o_similarity[:, 1] = similarity[:]
#
#             return o_similarity
#
#         return o_score
###########################################################################
