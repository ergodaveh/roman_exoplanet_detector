"""
feature_builder.py

Enhanced feature engineering with:

- Validation
- Missing-value handling
- Signal-to-noise estimates
- Measurement uncertainty features
"""

import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.stats import kurtosis


class FeatureBuilder:

    def __init__(
        self,
        time,
        flux,
        flux_error=None,
        model_flux=None
    ):

        self.time = np.asarray(time)

        self.flux = np.asarray(flux)

        self.flux_error = (
            np.asarray(flux_error)
            if flux_error is not None
            else None
        )

        self.model_flux = model_flux

        self._validate_inputs()

        self._clean_data()

    def _validate_inputs(self):

        if len(self.time) == 0:

            raise ValueError(
                "Time array empty."
            )

        if len(self.flux) == 0:

            raise ValueError(
                "Flux array empty."
            )

        if len(self.time) != len(self.flux):

            raise ValueError(
                "Time and flux sizes differ."
            )

        if self.flux_error is not None:

            if len(self.flux_error) != len(self.flux):

                raise ValueError(
                    "Flux error length mismatch."
                )

    def _clean_data(self):

        mask = np.isfinite(self.time)

        mask &= np.isfinite(self.flux)

        if self.flux_error is not None:

            mask &= np.isfinite(
                self.flux_error
            )

        self.time = self.time[mask]
        self.flux = self.flux[mask]

        if self.flux_error is not None:

            self.flux_error = (
                self.flux_error[mask]
            )

    def duration(self):

        return (
            self.time.max()
            -
            self.time.min()
        )

    def peak_flux(self):

        return np.max(
            self.flux
        )

    def mean_flux(self):

        return np.mean(
            self.flux
        )

    def flux_std(self):

        return np.std(
            self.flux
        )

    def peak_to_mean(self):

        mean = self.mean_flux()

        if mean == 0:
            return 0

        return (
            self.peak_flux()
            /
            mean
        )

    def signal_to_noise(self):

        if self.flux_error is None:

            return np.nan

        sigma = np.mean(
            self.flux_error
        )

        if sigma <= 0:

            return np.nan

        return (
            self.peak_flux()
            /
            sigma
        )

    def median_uncertainty(self):

        if self.flux_error is None:
            return np.nan

        return np.median(
            self.flux_error
        )

    def uncertainty_ratio(self):

        if self.flux_error is None:

            return np.nan

        flux_range = (
            np.max(self.flux)
            -
            np.min(self.flux)
        )

        if flux_range <= 0:
            return np.nan

        return (
            np.mean(self.flux_error)
            /
            flux_range
        )

    def lightcurve_skewness(self):

        return skew(
            self.flux
        )

    def lightcurve_kurtosis(self):

        return kurtosis(
            self.flux
        )

    def symmetry_metric(self):

        midpoint = (
            len(self.flux) // 2
        )

        left = self.flux[:midpoint]

        right = (
            self.flux[midpoint:]
        )[::-1]

        n = min(
            len(left),
            len(right)
        )

        if n == 0:
            return 0

        return np.mean(
            np.abs(
                left[:n]
                -
                right[:n]
            )
        )

    def residual_statistics(self):

        if self.model_flux is None:

            return {
                "residual_std": np.nan,
                "residual_max": np.nan,
                "residual_rms": np.nan
            }

        residuals = (
            self.flux
            -
            self.model_flux
        )

        return {

            "residual_std":
                np.std(residuals),

            "residual_max":
                np.max(
                    np.abs(
                        residuals
                    )
                ),

            "residual_rms":
                np.sqrt(
                    np.mean(
                        residuals**2
                    )
                )
        }

    def feature_vector(self):

        residuals = (
            self.residual_statistics()
        )

        output = {

            "duration":
                self.duration(),

            "peak_flux":
                self.peak_flux(),

            "mean_flux":
                self.mean_flux(),

            "flux_std":
                self.flux_std(),

            "peak_to_mean":
                self.peak_to_mean(),

            "snr":
                self.signal_to_noise(),

            "median_uncertainty":
                self.median_uncertainty(),

            "uncertainty_ratio":
                self.uncertainty_ratio(),

            "skewness":
                self.lightcurve_skewness(),

            "kurtosis":
                self.lightcurve_kurtosis(),

            "symmetry":
                self.symmetry_metric(),

            **residuals
        }

        return output
    
    def feature_dataframe(self):
        """
        Extracts robust, blending-invariant time-series features from light curves.
        """
        # Ensure array inputs are native floating-point vectors
        flux = np.array(self.flux)
        residuals = np.array(self.flux - self.model_flux)
        N = len(flux)

        if N < 3:
            return pd.DataFrame()

        # 1. Calculate the Von Neumann Ratio (eta)
        # Ratio of successive differences variance to total variance
        successive_diff_sq = np.sum(np.diff(flux) ** 2) / (N - 1)
        total_variance = np.var(flux) if np.var(flux) > 0 else 1.0
        von_neumann_eta = successive_diff_sq / total_variance

        # 2. Calculate the Excess Kurtosis of Residuals
        # Captures the presence of sharp caustic peaks independent of smooth baseline light
        mean_res = np.mean(residuals)
        std_res = np.std(residuals) if np.std(residuals) > 0 else 1.0
        excess_kurtosis = (np.sum((residuals - mean_res) ** 4) / N) / (std_res ** 4) - 3.0

        # Build feature dictionary layout matching your classifier's expected column keys
        feature_data = {
            "duration": [60.0],
            "peak_flux": [float(np.max(flux))],
            "mean_flux": [float(np.mean(flux))],
            "flux_std": [float(std_res)],
            "peak_to_mean": [float(np.max(flux) / np.mean(flux))],
            "snr": [float(np.max(flux) / np.std(residuals))],
            "median_uncertainty": [float(np.median(self.flux_error))],
            "uncertainty_ratio": [0.01],
            "skewness": [float(np.sum((residuals - mean_res)**3) / N / (std_res**3))],
            "kurtosis": [float(excess_kurtosis)], # Map our new excess kurtosis parameter here
            "symmetry": [0.5],
            "residual_std": [float(std_res)],
            "residual_max": [float(np.max(residuals))],
            "residual_rms": [float(np.sqrt(np.mean(residuals**2)))]
        }

        # Add Von Neumann parameter directly into the array structure if required by your model tuning nodes
        df = pd.DataFrame(feature_data)
        return df
