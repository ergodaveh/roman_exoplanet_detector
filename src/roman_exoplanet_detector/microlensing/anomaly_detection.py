"""
anomaly_detection.py
"""

import numpy as np

from scipy.signal import find_peaks


class AnomalyDetector:

    def residuals(

        self,

        flux,

        model

    ):

        return flux - model

    def detect(

        self,

        flux,

        model,

        threshold_sigma=5

    ):

        residuals = (
            self.residuals(
                flux,
                model
            )
        )

        sigma = np.std(
            residuals
        )

        peaks, properties = (

            find_peaks(

                np.abs(residuals),

                height=
                threshold_sigma
                *
                sigma
            )
        )

        return {

            "candidate":
                len(peaks) > 0,

            "num_peaks":
                len(peaks),

            "peak_locations":
                peaks,

            "residuals":
                residuals
        }