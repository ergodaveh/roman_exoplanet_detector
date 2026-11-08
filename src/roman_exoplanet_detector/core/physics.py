"""
physics.py

Astrophysical engine with analytical error propagation for converting 
dimensionless light curve parameters into absolute physical quantities.
"""
import numpy as np

# Astrophysical Constants
G = 6.67430e-11          # m^3 kg^-1 s^-2
C = 299792458            # m/s
M_SUN = 1.98847e30        # kg
M_EARTH = 5.9722e24       # kg
AU = 1.495978707e11      # meters
PC = 3.085677581e16      # parsec (meters)
SUN_EARTH_RATIO = M_SUN / M_EARTH

def calculate_physical_system_with_errors(
    mass_ratio_q: float,
    sigma_q: float,
    separation_s: float,
    sigma_s: float,
    apparent_mag: float,
    sigma_mag: float = 0.05,
    d_lens_pc: float = 6500.0,
    sigma_dl_pc: float = 500.0,
    d_source_pc: float = 8000.0
) -> dict:
    """
    Converts fitted dimensions to Earth masses, Solar masses, and AU 
    while propagating observational and modeling uncertainties using the Delta Method.
    """
    extinction = 1.5
    
    # 1. Absolute Magnitude & Variance
    abs_mag = apparent_mag - 5 * np.log10(d_lens_pc) + 5 - extinction
    # Derivative w.r.t d_lens: -5 / (d_lens * ln(10))
    dabs_ddl = -5.0 / (d_lens_pc * np.log(10))
    sigma_abs_mag = np.sqrt(sigma_mag**2 + (dabs_ddl * sigma_dl_pc)**2)

    # 2. Host Mass & Variance (Piecewise Empirical Mass-Luminosity Scaling)
    if abs_mag < 4.0:
        exponent = (4.75 - abs_mag) / 4.0
        host_m = 10 ** exponent
        dhost_dabs = -0.25 * host_m * np.log(10)
    elif abs_mag <= 11.0:
        exponent = (4.65 - abs_mag) / 3.5
        host_m = 10 ** exponent
        dhost_dabs = -(1.0 / 3.5) * host_m * np.log(10)
    else:
        exponent = (5.5 - abs_mag) / 2.3
        host_m = 10 ** exponent
        dhost_dabs = -(1.0 / 2.3) * host_m * np.log(10)
        
    sigma_host_m = abs(dhost_dabs * sigma_abs_mag)

    # 3. Planet Mass & Variance
    planet_m_earth = mass_ratio_q * host_m * SUN_EARTH_RATIO
    # Delta method components
    dp_dq = host_m * SUN_EARTH_RATIO
    dp_dhost = mass_ratio_q * SUN_EARTH_RATIO
    sigma_planet_m = np.sqrt((dp_dq * sigma_q)**2 + (dp_dhost * sigma_host_m)**2)

    # 4. Einstein Radius & Variance
    dl_m = d_lens_pc * PC
    ds_m = d_source_pc * PC
    
    # Base angular components
    theta_E_factor = (4 * G * M_SUN / (C**2)) * ((ds_m - dl_m) / (dl_m * ds_m))
    r_E_au = (np.sqrt(host_m * theta_E_factor) * dl_m) / AU
    
    # Derivative of r_E w.r.t host_m is 0.5 * r_E / host_m
    drE_dhost = 0.5 * r_E_au / host_m
    sigma_re = abs(drE_dhost * sigma_host_m)

    # 5. Projected Separation & Variance
    projected_sep_au = separation_s * r_E_au
    dsep_ds = r_E_au
    dsep_dre = separation_s
    sigma_sep = np.sqrt((dsep_ds * sigma_s)**2 + (dsep_dre * sigma_re)**2)

    return {
        "host_mass": float(host_m),
        "sigma_host_mass": float(sigma_host_m),
        "planet_mass": float(planet_m_earth),
        "sigma_planet_mass": float(sigma_planet_m),
        "separation": float(projected_sep_au),
        "sigma_separation": float(sigma_sep),
        "einstein_radius_au": float(r_E_au)
    }
