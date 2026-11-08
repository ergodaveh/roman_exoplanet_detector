"""
Microlensing model.
"""

import numpy as np


def amplification(u):

    return (
        (u**2 + 2)
        /
        (u * np.sqrt(u**2 + 4))
    )


def paczynski(t,
              t0,
              u0,
              tE,
              fs=1.0):

    u = np.sqrt(
        u0**2
        +
        ((t - t0) / tE)**2
    )

    return fs * amplification(u)