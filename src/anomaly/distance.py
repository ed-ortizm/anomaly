"""
Module with functionality to compute anomaly scores based on reconstructions
"""
from collections import namedtuple
import sys

import numpy as np

from anomaly.metrics import Distance
from anomaly.utils import spectra_to_batch_image, VelocityFilter


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
        observation = spectra_to_batch_image(observation)

        assert observation.ndim == 4

        reconstruction = self.reconstruct(observation[:, 0, :, 0])

        if self.filter_lines is True:

            observation = self.filter_object.filter(observation[:, 0, :, 0])
            reconstruction = self.filter_object.filter(reconstruction)

        else:

            observation = observation[:, 0, :, 0]

        if metric == "correlation":

            anomaly_score = super().correlation(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        if metric == "cosine":

            anomaly_score = super().cosine(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        if metric == "braycurtis":

            anomaly_score = super().braycurtis(observation, reconstruction)
            return anomaly_score.reshape((-1, 1))

        print(f"{metric} not implemented")
        sys.exit()
