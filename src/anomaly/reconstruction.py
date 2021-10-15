import sys
import numpy as np

################################################################################
class ReconstructionAnomalyScore:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    ############################################################################
    def __init__(self,
        model: "tf.keras.model"
        ):
        """
        INPUTS
            model: trained generative model with reconstruct method
        OUTPUT
            object
        """
        self.model = model
    ###########################################################################
    def _reconstruct(self,
        observation: "numpy array",
        ):

        return self.model.reconstruct(observation)
    ###########################################################################
    def _update_dimensions(self, x: "np.array") -> "numpy array":

        if x.ndim == 1:
            x = x[np.newaxis, ...]

        return x

    ###########################################################################
    def _get_reconstruction_error_ids(self,
        flux_wise_error: "numpy array",
        percentage: "int",
        )->"numpy array":

        """
        PARAMETERS
            flux_wise_error: array with reconstruction errors
                flux by flux
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
        OUTPUT
            anomaly score of the input observation
        """

        number_fluxes = flux_wise_error.shape[1]
        number_anomalous_fluxes = int(0.01 * percentage * number_fluxes)

        largest_reconstruction_error_ids = np.argpartition(
            flux_wise_error,
            -1 * number_anomalous_fluxes,
            axis=1)[:, -1 * number_anomalous_fluxes:]

        return largest_reconstruction_error_ids
    ###########################################################################
    def _get_anomaly_score(
        self, flux_wise_error: "numpy_array", percentage: "int"
    ) -> "numpy array":

        """
        PARAMETERS
            flux_wise_error: array with reconstruction errors
                flux by flux
            percentage: percentage of fluxes with the highest
                reconstruction error to consider to compute
                the anomaly score
        OUTPUT
            anomaly score of the input observation
        """

        reconstruction_error_ids = self._get_reconstruction_error_ids(
            flux_wise_error,
            percentage,
        )

        anomaly_score = np.empty(reconstruction_error_ids.shape)

        for n, idx in enumerate(reconstruction_error_ids):

            anomaly_score[n, :] = flux_wise_error[n, idx]

        return np.mean(anomaly_score, axis=1)

    ###########################################################################
    def mse(
        self,
        observation: "numpy array",
        percentage: "int",
        relative: "bool" = False,
        epsilon: "float" = 1e-3,
        )-> "numpy array":

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
        flux_wise_error = np.square(reconstruction - observation)

        if relative:
            flux_wise_error *= 1./np.square(observation + epsilon)

        flux_wise_error = self._update_dimensions(flux_wise_error)

        anomaly_score = self._get_anomaly_score(flux_wise_error, percentage)

        return anomaly_score
    ###########################################################################
#     def _mad(self, O, R, percentage, image):
#
#         reconstruction = np.abs(R - O)
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
#         o_score = score.sum(axis=1)
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
#     def _mad_relative(self, O, R, percentage, image):
#
#         if image:
#             O, R = O[:, 0, :, 0], R[:, 0, :, 0]
#
#         reconstruction = np.abs( (R - O)/(O + self.epsilon) )
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
#         o_score = np.sum(score, axis=1)
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
#     ############################################################################
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
#     ############################################################################
#     ############################################################################
#     def top_reconstructions(self, scores, n_top_spectra):
#         """
#         Selects the most normal and outlying objecs
#
#         Args:
#             scores: (1D np.array) outlier scores
#
#             n_top_spectra: (int > 0) this parameter controls the number of
#                 objects identifiers to return for the top reconstruction,
#                 that is, the idices for the most oulying and the most normal
#                 objects.
#
#         Returns:
#             most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
#                 with the location indexes of the most normal and outlying
#                 object in the training (and pred) set.
#         """
#
#         spec_idxs = np.argpartition(scores,
#             [n_top_spectra, -1 * n_top_spectra])
#
#         most_normal_ids = spec_idxs[: n_top_spectra]
#         most_oulying_ids = spec_idxs[-1 * n_top_spectra:]
#
#         return most_normal_ids, most_oulying_ids
# ################################################################################
#     # def _mse(self, O, R, percentage, image):
#     #
#     #     if image:
#     #         O, R = O[:, 0, :, 0], R[:, 0, :, 0]
#     #
#     #     mse = np.abs(R - O)/np.abs(O+0.0001)
#     #
#     #     number_outlier_fluxes = int(percentage*mse.shape[1])
#     #     highest_mse = np.argpartition(mse, -1 * number_outlier_fluxes,
#     #         axis=1)[:, -1 * number_outlier_fluxes:]
#     #
#     #     score = np.empty(highest_mse.shape)
#     #
#     #     for n, idx in enumerate(highest_mse):
#     #
#     #         score[n, :] = mse[n, idx]
#     #
#     #     o_score = score.sum(axis=1)
#     #
#     #     if image:
#     #
#     #         similarity =  o_score.max() - o_score
#     #
#     #         o_similarity = np.empty((o_score.size, 2))
#     #         o_similarity[:, 0] = o_score[:]
#     #         o_similarity[:, 1] = similarity[:]
#     #
#     #         return o_similarity
#     #
#     #     return o_score
