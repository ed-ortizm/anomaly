"""
Module with functionality to compute anomaly scores based on reconstructions
"""
from collections import namedtuple
import sys

import numpy as np

from anomaly.metrics import Reconstruction
from anomaly.utils import VelocityFilter, spectra_to_batch_image


class ReconstructionAnomalyScore(Reconstruction):
    """
    Class to deal with the outliers based on a generative model
    trained with tensorflow.keras. The class uses metrics based
    on reconstruction residuals
    """

    def __init__(
        self,
        reconstruct_function,
        reconstruction_parameters: namedtuple,
        filter_parameters: namedtuple,
    ):
        """
        INPUTS
            reconstruct_function: reconstruct method of trained
                generative model

            reconstruction_parameters: named tuple containing
                percentage: percentage of fluxes with the highest
                    reconstruction error to consider to compute
                    the anomaly score
                relative: whether or not the score is weigthed by
                    the input
                epsilon: factor to add for numerical stability
                    when dividing by the input spectra

            filter_parameters: named tuple containing
                lines: list with lines to discard to compute
                    anomaly_score
                velocity_filter: Doppler velocity to consider at
                    the moment of line filtering. It is in units
                    of km/s. DeltaWave = (v/c) * wave
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

        super().__init__(
            percentage=reconstruction_parameters.percentage,
            relative=reconstruction_parameters.relative,
            epsilon=reconstruction_parameters.epsilon
        )

    def score(
        self, observation: np.array, metric: str, p: float = 0.33
    ) -> np.array:
        """Compute reconstruction error """

        # in case I pass a spectra with one dimension
        # this line converts 1D array to (1, n_wave, 3)
        # an image where each channel has the spectrun
        # for compatibility with lime image explainer :)
        observation = spectra_to_batch_image(observation)

        assert observation.ndim == 4

        reconstruction = self.reconstruct(observation[:, 0, :, 0])

        if self.filter_lines is True:

            observation = self.filter_object.filter(observation[:, 0, :, 0])
            reconstruction = self.filter_object.filter(reconstruction)

        # observation as image to obserbation as spectra
        else:

            observation = observation[:, 0, :, 0]

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

        if metric == "braycurtis":

            anomaly_score = super().braycurtis(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        print(f"{metric} not implemented")
        sys.exit()
