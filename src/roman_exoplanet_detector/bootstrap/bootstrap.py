"""
bootstrap.py

Creates required resources if missing.

Resources:
- test_roman.asdf
- classifier.joblib
"""

from pathlib import Path

import asdf
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier
)


class ProjectBootstrap:

    def __init__(self):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[4]
        )

        self.data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

        self.model_dir = (
            self.project_root
            / "models"
        )

        self.data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    
        # --------------------------------------------------
    # Multi-Epoch Multi-Band Roman ASDF Stack Generator
    # --------------------------------------------------

    def create_test_asdf(
        self,
        base_filename 
    ):
        base_path = Path(base_filename)
        data_directory = base_path.parent
        data_directory.mkdir(parents=True, exist_ok=True)

        print("Generating professional multi-epoch 2D image stack (100 chronological epochs)...")

        # 1. Define our temporal framework (100 sampling epochs across a 60-day campaign)
        epochs = 100
        time_steps = np.linspace(0.0, 60.0, epochs)

        # 2. Relativistic Microlensing Model configuration parameters for the Source 1 variable star
        t0, tE, u0 = 30.0, 22.5, 0.15
        blending_fraction_fb = 0.75
        t_anomaly = 34.2

        # 3. Base 2D image layout and static star catalog definitions
        shape = (256, 256)
        y_indices, x_indices = np.indices(shape)
        sigma_psf = 1.5

        # Format: (Y_center, X_center, Baseline_Flux_W146, Color_Profile_Type)
        star_catalog = [
            (50.0, 50.0, 1500.0, 'COOL_RED_DWARF'),    # Source 1: Dynamic Microlensing Target
            (120.0, 140.0, 2500.0, 'STELLAR_NORMAL'),  # Source 2: Static Baseline Check Star
            (180.0, 75.0, 1800.0, 'HOT_BLUE'),         # Source 3: Static Baseline Check Star
            (220.0, 200.0, 3000.0, 'COOL_RED_DWARF'),  # Source 4: Static Baseline Check Star
            (100.0, 220.0, 2100.0, 'STELLAR_NORMAL'),  # Source 5: Static Baseline Check Star
        ]

        # 4. Generate the sequential frame stack file-by-file
        for frame_idx, t in enumerate(time_steps):
            # Compute dynamic magnification factor for Source 1 at this exact timestamp (t)
            u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
            magnification_A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
            
            # --- UPGRADED: TRUE ASYMMETRIC CAUSTIC FOLD CROSSING PROFILE ---
            # Replaces the old smooth symmetric Gaussian envelope profile shape
            if t < t_anomaly:
                planetary_anomaly = 0.0
            else:
                # Implements the characteristic step-jump followed by a 1/sqrt(dt) geometric decay.
                # A softening value of 0.05 prevents division-by-zero numerical errors at the spike.
                dt = t - t_anomaly
                planetary_anomaly = 0.35 / np.sqrt(dt + 0.05) * np.exp(-dt / 4.0)
            
            total_magnification = magnification_A + planetary_anomaly
            # ----------------------------------------------------------------
            
            # Calculate the current scaled magnification ratio
            magnification_ratio = (blending_fraction_fb * total_magnification + (1.0 - blending_fraction_fb))

            # Initialize fresh noise planes for this epoch's detectors
            image_w146 = np.random.normal(10.0, 1.0, shape)
            image_f087 = np.random.normal(4.0, 0.5, shape)
            image_f184 = np.random.normal(5.0, 0.6, shape)

            # Paint all 5 stars onto the 2D arrays
            for s_idx, (y_center, x_center, base_flux, star_type) in enumerate(star_catalog):
                r_sq = (x_indices - x_center)**2 + (y_indices - y_center)**2
                gaussian_psf_profile = np.exp(-r_sq / (2 * (sigma_psf**2)))

                # If processing Source 1, apply the time-dependent magnification scalar
                current_base_flux = base_flux * magnification_ratio if s_idx == 0 else base_flux

                if star_type == 'COOL_RED_DWARF':
                    flux_f087 = current_base_flux * 0.25
                    flux_f184 = current_base_flux * 1.40
                elif star_type == 'HOT_BLUE':
                    flux_f087 = current_base_flux * 1.50
                    flux_f184 = current_base_flux * 0.30
                else:
                    flux_f087 = current_base_flux * 0.85
                    flux_f184 = current_base_flux * 0.90

                image_w146 += current_base_flux * gaussian_psf_profile
                image_f087 += flux_f087 * gaussian_psf_profile
                image_f184 += flux_f184 * gaussian_psf_profile

            # 5. Package this individual epoch frame into a structured ASDF tree
            tree = {
                "data": image_w146,
                "err": np.full(shape, 1.0),
                "dq": np.zeros(shape, dtype=np.uint16),
                "roman_wfi_bands": {
                    "W146": {"data": image_w146, "err": np.full(shape, 1.0)},
                    "F087": {"data": image_f087, "err": np.full(shape, 0.5)},
                    "F184": {"data": image_f184, "err": np.full(shape, 0.6)}
                },
                "meta": {
                    "instrument": {"optical_element": "MULTI-BAND"},
                    "observation": {
                        "start_time": "2027-01-01T00:00:00",
                        "epoch_time_days": float(t),
                        "frame_index": int(frame_idx)
                    }
                }
            }

            # Write out chronologically numbered file assets (e.g., roman_file_000.asdf)
            frame_filename = data_directory / f"{base_path.stem}_{frame_idx:03d}.asdf"
            af = asdf.AsdfFile(tree)
            af.write_to(frame_filename)

        # Write out a zero-indexed fallback file to preserve current main pipeline paths
        # This keeps the main.py execution running without a directory loader refactor
        af_fallback = asdf.AsdfFile(tree)
        af_fallback.write_to(base_path)
        print(f"Successfully generated a 100-frame image stack inside: {data_directory}")


    # ----------------------------------
    # Classifier
    # ----------------------------------

    def create_classifier(
        self,
        filename 
    ):
        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        print(
            "Creating bootstrap model..."
        )

        rows = []

        np.random.seed(42)

        for _ in range(1000):

            rows.append({

                "duration":
                    np.random.uniform(10, 100),

                "peak_flux":
                    np.random.uniform(1, 5),

                "mean_flux":
                    np.random.uniform(0.8, 2.0),

                "flux_std":
                    np.random.uniform(0.01, 0.5),

                "peak_to_mean":
                    np.random.uniform(1.0, 5.0),

                "snr":
                    np.random.uniform(5, 100),

                "median_uncertainty":
                    np.random.uniform(0.001, 0.05),

                "uncertainty_ratio":
                    np.random.uniform(0.001, 0.2),

                "skewness":
                    np.random.uniform(-2, 2),

                "kurtosis":
                    np.random.uniform(-1, 5),

                "symmetry":
                    np.random.uniform(0, 1),

                "residual_std":
                    np.random.uniform(0, 0.1),

                "residual_max":
                    np.random.uniform(0, 0.2),

                "residual_rms":
                    np.random.uniform(0, 0.1),

                "label":
                    np.random.choice([0, 1]),
            })

        df = pd.DataFrame(
            rows
        )

        X = df.drop(
            columns=["label"]
        )

        y = df["label"]

        model = (
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )

        model.fit(
            X,
            y
        )

        joblib.dump(
            model,
            filename
        )

        print(
            f"Created {filename}"
        )

    # ----------------------------------
    # Verify
    # ----------------------------------

    def ensure_resources(
        self,
        roman_file,
        classifier_file
    ):

        roman_path = Path(
            roman_file
        )
        model_path = Path(
            classifier_file
        )
        print(
            f"Roman path = {roman_path.resolve()}"
        )

        print(
            f"Model path = {model_path.resolve()}"
        )

        if not roman_path.exists():

            self.create_test_asdf(
                roman_path
            )

        else:

            print(
                f"Found {roman_path}"
            )

        if not model_path.exists():

            self.create_classifier(
                model_path
            )

        else:

            print(
                f"Found {model_path}"
            )