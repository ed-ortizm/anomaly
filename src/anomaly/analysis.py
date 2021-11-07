import numpy as np


class AnalysisAnomalyScore:
    """"""
    ###########################################################################
    def __init__(self,
        anomaly_scores: "2D numpy array",
    ):
        """"""
        self.scores = anomaly_scores

    ###########################################################################
    def top_scores(self,
        number_normal: "int",
        number_anomalies: "int",
    )->"[1D numpy array, 1D numpy array]":
        """
        Find the most normal and most anomalous objecs

        PARAMETERS
            anomaly_scores: outlier scores
            number_normal: number of most normal objects
            number_anomalies: number of most anomalous objects


        Returns:
            [normal_ids, anomalous_ids]: arrays with the location
                indexes of the most normal and anomalous objects.
        """


        normal_ids = self._get_top_ids(number_normal, anomalous=False)

        anomalous_ids = self._get_top_ids(number_anomalies, anomalous=True)

        return [normal_ids, anomalous_ids]
    ###########################################################################
    def _get_top_ids(
        self,
        number_objects: "int",
        anomalous:"bool",
    )->"1D numpy array":

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

        score_ids = np.argpartition(
            self.scores,
            [number_objects, -1 * number_objects]
        )
        if anomalous:
            objects_ids = score_ids[-1 * number_objects:]
        else:
            objects_ids = score_ids[: number_objects]

        return objects_ids
