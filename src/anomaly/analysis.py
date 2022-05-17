"""Module with utilities to analyse anomalous spectra"""
import numpy as np


class AnalysisAnomalyScore:
    """Analysis of scores"""

    ###########################################################################
    def __init__(self, anomaly_scores: np.array):
        """"""
        self.scores = anomaly_scores

    ###########################################################################
    def get_percentiles(
        self, range_of_percentiles: list = None
    ) -> np.array:

        """
        Percentiles of anomaly scores. 10 means it will return the 10% of
        anomaly scores close to zero
        """

        if range_of_percentiles is None:
            range_of_percentiles = [0, 25, 55, 75, 99.9, 100]

        return np.percentile(self.scores, q=range)

    ###########################################################################
    def top_scores(
        self, number_scores: int, anomalous: bool = True
    ) -> np.array:
        """
        Find the most normal and most anomalous objecs

        PARAMETERS

        OUTPUTS
        """

        scores_ids = self._get_top_ids(number_scores, anomalous)

        return scores_ids

    ###########################################################################
    def _get_top_ids(self, number_objects: int, anomalous: bool) -> np.array:

        """
        Get the ids with the objects having the largest or smallest
            anomaly scores depending on anomlaous parameter

        PARAMETERS
            number_objects: number of indexes to retrieve
                from scores array
            anomalous: If True it returns the ids of the objects with
                the highest anomaly scores, if False then it return
                the ids of the objects with the smallest anomaly scores

        OUTPUTS
            objects_ids: ids of objects in the anomaly score array

        """

        if anomalous:
            number_objects = -1 * number_objects

        scores_ids = np.argpartition(self.scores, number_objects)

        if anomalous:
            return scores_ids[number_objects:]

        return scores_ids[:number_objects]
