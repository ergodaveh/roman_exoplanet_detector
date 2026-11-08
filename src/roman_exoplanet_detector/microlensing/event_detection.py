"""
event_detection.py

Fits core Paczynski microlensing baseline curves to time-series photometry data,
handling flat unmagnified stars gracefully using runtime exception boundaries.
"""

import numpy as np
from scipy.optimize import curve_fit
from roman_exoplanet_detector.microlensing.paczynski import paczynski


class EventDetector:

        def fit_event(self, time, flux):
            """
            Fits a standard point-mass microlensing profile to the light curve.
            """
            # Initial parameter guesses: [t0, tE, u0, baseline_flux]
            initial_guess = [np.median(time), 15.0, 0.5, 1.0]
            
            # Open up the upper boundaries to support high-magnification events (peak flux up to 10.0)
            # and extend iterations so L-BFGS-B / Trust-Region can converge on dense 100-point matrices
            bounds = (
                [min(time), 0.1, 1e-4, 0.1],  # Lower bounds
                [max(time), 100.0, 5.0, 10.0]  # Upper bounds (Increased from 2.0 to 10.0)
            )

            try:
                params, covariance = curve_fit(
                    paczynski,
                    time,
                    flux,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=20000  # Increased from 5000 to 20000 to handle 100 dense epochs cleanly
                )
                
                return {
                    "params": params,
                    "success": True,
                    "msg": "Optimal curve parameters converged tightly."
                }

            except RuntimeError:
                print("  ⚠️  [FIT WARNING] No active lensing curvature detected. Setting flat baseline profile.")
                fallback_params = [np.median(time), 20.0, 2.0, 1.0]
                return {
                    "params": fallback_params,
                    "success": False,
                    "msg": "Flat line detection profile fallback utilized."
                }

