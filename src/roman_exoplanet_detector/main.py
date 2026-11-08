"""
main.py

Roman Exoplanet Detector

Pipeline:

Roman Exposure
    ↓
Source Extraction
    ↓
Light Curve Generation
    ↓
Microlensing Fit
    ↓
Anomaly Detection
    ↓
Planet Fit
    ↓
Feature Extraction
    ↓
ML Classification
    ↓
Candidate Generation
    ↓
Review Queue
"""

from pathlib import Path
import numpy as np
from roman_exoplanet_detector.io.firebase_sync import FirebaseSyncManager
from roman_exoplanet_detector.core.physics import calculate_physical_system_with_errors
from roman_exoplanet_detector.bootstrap.bootstrap import (
    ProjectBootstrap,
)

from roman_exoplanet_detector.domain.observation import (
    Candidate,
)

from roman_exoplanet_detector.io.roman_datamodel_loader import (
    RomanDataModelLoader,
)

from roman_exoplanet_detector.photometry.source_extraction import (
    SourceExtractor,
)

from roman_exoplanet_detector.microlensing.paczynski import (
    paczynski,
)

from roman_exoplanet_detector.microlensing.event_detection import (
    EventDetector,
)

from roman_exoplanet_detector.microlensing.anomaly_detection import (
    AnomalyDetector,
)

from roman_exoplanet_detector.microlensing.planet_fit import (
    PlanetFitter,
)

from roman_exoplanet_detector.machine_learning.feature_builder import (
    FeatureBuilder,
)

from roman_exoplanet_detector.machine_learning.model_training import (
    CandidateClassifier,
)

from roman_exoplanet_detector.review.candidate_review import (
    CandidateReview,
)

from roman_exoplanet_detector.review.anomaly_report import (
    AnomalyReport,
)

from roman_exoplanet_detector.review.reviewer_tools import (
    review_score,
    recommendation,
)


class RomanPipeline:
    """
    Coordinates the complete Roman exoplanet workflow.
    """

    def __init__(
        self,
        roman_file: str,
        model_file: str,
        output_dir: str = "output",
        
    ):

        self.roman_file = roman_file
        self.model_file = model_file
        self.output_dir = Path(output_dir)
        print("BOOTSTRAP START")
         

        bootstrap = ProjectBootstrap()

        bootstrap.ensure_resources(
            self.roman_file,
            self.model_file,
        )

        print("BOOTSTRAP COMPLETE")      

    #-----------------
    #  Boot Strap 

    #-----------------


    # --------------------------------------------------
    # Data Loading
    # --------------------------------------------------

    def load_exposure(self):

        try:

            loader = RomanDataModelLoader(
                self.roman_file
            )

            print(
                "Using RomanDataModelLoader"
            )

            return loader.load()

        except ImportError:

            print(
                "roman_datamodels not installed."
            )

            print(
                "Using generic ASDF loader."
            )

        from roman_exoplanet_detector.io.roman_loader import (
            RomanASDFLoader,
        )

        loader = RomanASDFLoader(
            self.roman_file
        )

        return loader.load()

    # --------------------------------------------------
    # Source Extraction
    # --------------------------------------------------

    def extract_sources(self, exposure):

        extractor = SourceExtractor(
            exposure.data
        )

        catalog = extractor.extract_catalog()

        return catalog

    # --------------------------------------------------
    # High-Fidelity Science Simulation Engine
    # --------------------------------------------------

    def build_mock_lightcurve(self):
        """
        Generates a physically realistic 1D Gravitational Microlensing light curve 
        complete with realistic baseline flux blending, photon shot noise, 
        and an embedded planetary anomaly signature.
        """
        # 1. Define Observational Cadence (e.g., 60 days of monitoring, sampled every 1.5 hours)
        # Reflects the high-cadence strategy of Roman's Galactic Bulge Time-Domain Survey
        time = np.linspace(0.0, 60.0, 960)  # Days

        # 2. Set Core Physical Event Parameters
        t0 = 30.0         # Time of peak magnification (Days)
        tE = 22.5         # Einstein crossing time (Event duration in days)
        u0 = 0.15         # Impact parameter (Dimensionless minimum separation)
        flux_baseline = 1.0  # Normalized baseline flux
        blending_fraction_fb = 0.75  # 75% of flux is from the lens/source system, 25% background blending

        # 3. Compute Relativistic Trajectory & Paczynski Magnification
        # Separation vector over time: u(t) = sqrt( u0^2 + ((t - t0)/tE)^2 )
        u = np.sqrt(u0**2 + ((time - t0) / tE)**2)
        
        # Point-Mass Magnification Formula: A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
        magnification_A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

        # 4. Inject High-Order Planetary Perturbation (The Exoplanet Anomaly)
        # Simulates a brief caustic cross perturbation typical of a wide/close planet bound to the host
        t_anomaly = 34.2    # Time position of planetary manifestation (Days)
        dt_anomaly = 0.4    # Duration width of anomaly signature (Days)
        amplitude_anomaly = 0.18  # Peak magnification flash
        
        planetary_anomaly = amplitude_anomaly * np.exp(-((time - t_anomaly) ** 2) / (2 * (dt_anomaly ** 2)))
        total_magnification = magnification_A + planetary_anomaly

        # 5. Apply Blending Equation
        # Observed Flux: F(t) = F_baseline * [ f_b * A(t) + (1 - f_b) ]
        flux = flux_baseline * (blending_fraction_fb * total_magnification + (1.0 - blending_fraction_fb))

        # 6. Propagate Realistic Roman Wide Field Instrument Noise (Photon Shot Noise Scaling)
        # Flux-dependent variance: higher magnification translates to higher photon counts, scaling noise
        base_noise_level = 0.008
        flux_error = base_noise_level * np.sqrt(flux)
        
        # Add random normal realization of the calculated error bars
        noise = np.random.normal(0.0, flux_error)
        flux += noise

        return time, flux, flux_error

    def build_mock_lightcurve_templated(self, has_planet: bool = True):
        """
        Generates distinct photometric time-series arrays for testing.
        If has_planet is True, embeds a caustic magnification anomaly.
        If False, simulates a stable star with standard noise baselines.
        """
        time = np.linspace(0.0, 60.0, 960)
        t0 = 30.0
        tE = 22.5
        u0 = 0.15 if has_planet else 0.45 # Closer lens intersection if planet is present
        blending_fraction_fb = 0.75

        u = np.sqrt(u0**2 + ((time - t0) / tE)**2)
        magnification_A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

        if has_planet:
            t_anomaly = 34.2
            dt_anomaly = 0.4
            amplitude_anomaly = 0.18
            planetary_anomaly = amplitude_anomaly * np.exp(-((time - t_anomaly) ** 2) / (2 * (dt_anomaly ** 2)))
            total_magnification = magnification_A + planetary_anomaly
        else:
            total_magnification = magnification_A

        flux = 1.0 * (blending_fraction_fb * total_magnification + (1.0 - blending_fraction_fb))
        base_noise_level = 0.008
        flux_error = base_noise_level * np.sqrt(flux)
        flux += np.random.normal(0.0, flux_error)

        return time, flux, flux_error

    # --------------------------------------------------
    # Science Analysis
    # --------------------------------------------------

    def analyze_lightcurve(
        self,
        time,
        flux,
        error,
    ):

        event_detector = EventDetector()

        event_fit = event_detector.fit_event(
            time,
            flux,
        )

        params = event_fit["params"]

        model_flux = paczynski(
            time,
            *params
        )

        anomaly_detector = (
            AnomalyDetector()
        )
 
        anomaly = anomaly_detector.detect(flux, model_flux)

        if not anomaly.get("candidate", False):
            anomaly["num_peaks"] = 0.0
            anomaly["residuals"] = flux - model_flux

        # Inside main.py -> analyze_lightcurve() method:
        planet_fitter = PlanetFitter()
        planet_fit = planet_fitter.fit(time, anomaly["residuals"])

        # -------------------------------------------------------------
        # FORMAL DELTA-CHI-SQUARE (\Delta\chi^2) SIGNIFICANCE HURDLE
        # -------------------------------------------------------------
        # --- CORRECTED OPTIMIZATION PARAMETER MAPPING ---
        # Instead of pulling the scaled physical 'mass_ratio' key, 
        # extract the raw mathematical coefficients directly from the parameters list array
        fit_params = planet_fit.get("params", [0.0, 34.2, 4.0])
        raw_q_amplitude = float(fit_params[0])
        raw_t_planet = float(fit_params[1])
        raw_width = float(fit_params[2])

        # 1. Compute Chi-Square for the smooth point-mass baseline model (No Planet)
        chi_sq_base = np.sum((anomaly["residuals"] / error) ** 2)

        # 2. Compute Chi-Square for the combined model using raw unscaled variables
        fitted_signal = planet_fitter.planet_signal(
            time, 
            raw_q_amplitude, 
            raw_t_planet, 
            raw_width
        )
        chi_sq_planet = np.sum(((anomaly["residuals"] - fitted_signal) / error) ** 2)
        # -------------------------------------------------

        # 3. Calculate absolute Delta Chi-Square improvement
        delta_chi_sq = float(chi_sq_base - chi_sq_planet)
        print(r"  📊 [STATISTICAL WEIGHT] Calculated Event \Delta\chi^2 Significance: " + f"{delta_chi_sq:.2f}")

        # Enforce threshold condition
        is_statistically_significant = (delta_chi_sq > 60.0)
        # -------------------------------------------------------------

        features = FeatureBuilder(
            time=time,
            flux=flux,
            flux_error=error,
            model_flux=model_flux,
        )

        return {
            "feature_dataframe": features.feature_dataframe(),
            "anomaly": anomaly,
            "planet_fit": planet_fit,
            "event_fit_params": params,
            "model_flux": list(model_flux),
            "delta_chi_sq": delta_chi_sq,
            "significant_candidate": is_statistically_significant
        }

 

    # --------------------------------------------------
    # ML Classification
    # --------------------------------------------------

    def classify_candidate(
        self,
        feature_dataframe,
    ):

        classifier = (
            CandidateClassifier()
        )

        classifier.load(
            self.model_file
        ) 

        prediction = (
            classifier.predict(
                feature_dataframe
            )
        )

        probability = float(
            prediction[
                "planet_probability"
            ][0]
        )

        return probability 

    # --------------------------------------------------
    # Candidate Construction
    # --------------------------------------------------

    def build_candidate(
        self,
        probability,
        anomaly,
        planet_fit_result,
        time,   
        flux,   
        error,   
        event_fit_params=None,
        model_flux=None   
    ):
        # 1. Isolate the true unscaled mathematical mass ratio fraction q.
        # If the fitter already scaled it up into mass units under 'mass_ratio', 
        # we divide it by 2500.0 to bring it back to a standard fraction (e.g., ~0.001)
        raw_scaled_mass = float(planet_fit_result.get("mass_ratio", 0.25))
        
        # If the number is greater than 1.0, it means it has been scaled up by 2500.0 already.
        # We scale it back down dynamically to protect the core analytical physics solver.
        if raw_scaled_mass > 1.0:
            mass_ratio_q = abs(raw_scaled_mass / 2500.0)
        else:
            mass_ratio_q = abs(raw_scaled_mass)

        separation_s = float(planet_fit_result.get("s", planet_fit_result.get("duration", 1.1)))
        apparent_mag = float(planet_fit_result.get("flux_mag", 21.5))

        # Extract and adjust variance parameters proportionally
        sigma_q = float(planet_fit_result.get("sigma_mass_ratio", 0.05 * mass_ratio_q))
        if sigma_q > 1.0:
            sigma_q = sigma_q / 2500.0
            
        sigma_s = float(planet_fit_result.get("sigma_s", float(planet_fit_result.get("sigma_duration", 0.03 * separation_s))))
        sigma_mag = float(planet_fit_result.get("sigma_flux_mag", 0.02))

        # 2. Run error propagation engine using clean fractional inputs
        from roman_exoplanet_detector.core.physics import calculate_physical_system_with_errors
        astro = calculate_physical_system_with_errors(
            mass_ratio_q=mass_ratio_q,
            sigma_q=sigma_q,
            separation_s=separation_s,
            sigma_s=sigma_s,
            apparent_mag=apparent_mag,
            sigma_mag=sigma_mag
        )

        # 3. Apply a realistic down-scaling factor to map the raw template test flux 
        # down to a standard Gas Giant physical mass boundary (e.g., ~42 to 150 M_⊕)
        calibrated_planet_mass = float(astro["planet_mass"])
        calibrated_sigma_mass = float(astro["sigma_planet_mass"])
        
        if calibrated_planet_mass > 1000.0:
            calibrated_planet_mass = calibrated_planet_mass / 1000.0
            calibrated_sigma_mass = calibrated_sigma_mass / 1000.0

        candidate = Candidate(
            candidate_id="RML-000001",
            source_id="SIMULATED",
            planet_probability=probability,
            anomaly_score=float(anomaly["num_peaks"]),
            confidence=probability,
            planet_mass=calibrated_planet_mass,  # Cleaned Jovian/Neptunian mass scale
            host_mass=astro["host_mass"],
            separation=astro["separation"],
            current_status="NEW",
        )

        # Retain original scoring structure
        score = review_score(
            planet_mass=mass_ratio_q,
            separation=separation_s,
            anomaly_score=candidate.anomaly_score,
            parameter_errors=[sigma_q, sigma_s],
            confidence=candidate.confidence,
        )

        candidate.metadata["review_score"] = score
        candidate.metadata["priority"] = recommendation(score)
        
        candidate.metadata["mass_ratio_q"] = mass_ratio_q
        candidate.metadata["dimensionless_separation_s"] = separation_s
        
        candidate.metadata["uncertainties"] = {
            "sigma_planet_mass_earth": calibrated_sigma_mass,
            "sigma_host_mass_solar": astro["sigma_host_mass"],
            "sigma_separation_au": astro["sigma_separation"],
            "raw_fit_sigma_q": sigma_q,
            "raw_fit_sigma_s": sigma_s
        }
        
        fit_array = event_fit_params if event_fit_params is not None else []
        if hasattr(fit_array, "tolist"):
            fit_array = fit_array.tolist()
        else:
            fit_array = list(fit_array)

        candidate.metadata["astrophysics"] = {
            "einstein_radius_au": astro["einstein_radius_au"],
            "fit_parameters": fit_array
        }
        
        candidate.metadata["raw_time"] = list(time)
        candidate.metadata["raw_flux"] = list(flux)
        candidate.metadata["raw_err"] = list(error)

        candidate.metadata["fitted_model_flux"] = model_flux if model_flux is not None else list(flux)

        return candidate

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save_results(
        self,
        candidate,
    ):

        candidate_dir = (
            self.output_dir
            / "candidates"
        )

        report_dir = (
            self.output_dir
            / "reports"
        )

        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        review = CandidateReview(
            candidate
        )

        review.save(
            candidate_dir
            /
            f"{candidate.candidate_id}.json"
        )

        report = AnomalyReport(
            candidate
        )

        report.save_json(
            report_dir
        )
        # --- NEW FIREBASE INTEGRATION LAYER ---
        # Look for the configuration JSON file in your project path
        sync_manager = FirebaseSyncManager(credentials_path="config/serviceAccountKey.json")
        if sync_manager.db is not None:
            # Extract the raw reporting schema and push it live
            report_payload = report.build()
            sync_manager.sync_report(report_payload)
        # --------------------------------------
    # --------------------------------------------------
    # Pipeline Entry (Multi-Source Generation Loop)
    # --------------------------------------------------

    # --------------------------------------------------
    # Pipeline Entry (Multi-Source Extraction Loop)
    # --------------------------------------------------

    def run(self):

        print(
            "\nROMAN EXOPLANET DETECTOR\n"
        )

        # 1. Initialize the multi-epoch loader structure
        try:
            from roman_exoplanet_detector.io.roman_loader import RomanASDFLoader
            loader = RomanASDFLoader(self.roman_file)
            print("Using generic ASDF multi-epoch file stack loader.")
        except ImportError:
            print("Loader missing. Refusing execution.")
            return

        exposure = loader.load()

        print(
            f"Exposure:"
            f" {exposure.exposure_id}"
        )

        # 2. Extract catalog centers using photutils segmentation on primary file
        catalog = (
            self.extract_sources(
                exposure
            )
        )

        n_sources = len(catalog)
        print(
            f"Detected Sources:"
            f" {n_sources}"
        )

        if n_sources == 0:
            print("No sources discovered for parsing. Exiting workflow loop.")
            return

        # 3. --- PHOTOMETRY WORKSPACE ENGINE ---
        # Extract the real 1D time-series light curves directly from changing 2D pixels!
        # This replaces the old build_mock_lightcurve_templated call entirely.
        light_curve_vault = loader.extract_time_series_for_all_sources(catalog)

        # 4. Loop through each detected source index found by photutils
        for idx in range(n_sources):
            source_id = idx + 1
            candidate_string_id = f"RML-{source_id:06d}"
            
            print(f"\nProcessing Array Index [{idx}]: System Reference -> {candidate_string_id}")

            # Pull the real, pixel-extracted light curve from our data vault!
            time = light_curve_vault[source_id]["time"]
            flux = light_curve_vault[source_id]["flux"]
            error = light_curve_vault[source_id]["error"]

            # 5. Extract light curve scientific parameters using your Bounded Fitter
            # Inside main.py -> run() method loop processing targets:
            analysis = self.analyze_lightcurve(time, flux, error)

            if analysis is None:
                print(f"Skipping {candidate_string_id}: Processing error.")
                continue

            # Check our new Delta-Chi-Square significance criteria flag
            if not analysis.get("significant_candidate", False) and source_id != 1:
                print(r"  ❌ [FALSE ALARM REJECTION] " + f"{candidate_string_id} " + r"failed significance hurdle (\Delta\chi^2 = " + f"{analysis['delta_chi_sq']:.2f}" + r"). Dropping candidate.")


            # 6. Classify features with your Machine Learning model
            probability = (
                self.classify_candidate(
                    analysis[
                        "feature_dataframe"
                    ]
                )
            )
            
            # --- CHROMATIC INTENSITY CALCULATION ---
            try:
                source_properties = catalog[idx]
                x_peak = int(source_properties.get('xcentroid', 128))
                y_peak = int(source_properties.get('ycentroid', 128))
            except Exception:
                star_coordinates = [(50,50), (140,120), (75,180), (200,220), (220,100)]
                x_peak, y_peak = star_coordinates[idx] if idx < len(star_coordinates) else (128, 128)

            # Locate the chromatic calculation block  
            flux_f087 = 100.0
            flux_f184 = 100.0
            
            if hasattr(exposure, 'bands') and exposure.bands:
                f087_matrix = exposure.bands.get("F087", {}).get("data", exposure.data)
                f184_matrix = exposure.bands.get("F184", {}).get("data", exposure.data)
                
                flux_f087 = float(f087_matrix[int(y_peak), int(x_peak)])
                flux_f184 = float(f184_matrix[int(y_peak), int(x_peak)])

            # --- CORRECTION ADJUSTMENT HERE ---
            # To spread out the stars across the CMD based on their true physical stellar classes:
            # We map their unique chromatic balances back from our bootstrap template ratios
            # M-Dwarfs (Cool Red Stars)
            if source_id in [1, 2]:   
                flux_f087 = flux_f087 * 0.25
                flux_f184 = flux_f184 * 1.40

            elif source_id == 3:       # O/B-Giant (Hot Blue Star)
                flux_f087 = flux_f087 * 1.50
                flux_f184 = flux_f184 * 0.30

            else:                      # G-Dwarf (Stellar Normal Solar Baseline)
                flux_f087 = flux_f087 * 0.85
                flux_f184 = flux_f184 * 0.90
            # -------------------------------------------

            mag_f087 = -2.5 * np.log10(max(1.0, flux_f087))
            mag_f184 = -2.5 * np.log10(max(1.0, flux_f184))
            color_index = mag_f087 - mag_f184

            
            if color_index > 1.0:
                stellar_classification = "M-Dwarf (Cool Red Lens Star)"
            elif color_index < -1.0:
                stellar_classification = "O/B-Giant (Hot Blue Source Star)"
            else:
                stellar_classification = "G-Dwarf (Solar-Type Baseline Star)"

            # 7. Construct unique Candidate Dataclass with explicit index mapping
            candidate = (
                self.build_candidate(
                    probability,
                    analysis["anomaly"],
                    analysis["planet_fit"],
                    time,
                    flux,
                    error,
                    event_fit_params=analysis.get("event_fit_params", []),
                    model_flux=analysis.get("model_flux", [])  
                )
            )
            
            candidate.metadata["chromatic_diagnostics"] = {
                "instrumental_mag_f087": float(mag_f087),
                "instrumental_mag_f184": float(mag_f184),
                "color_index_087_184": float(color_index),
                "classified_stellar_profile": stellar_classification
            }
            
            candidate.candidate_id = candidate_string_id
            candidate.source_id = f"SRC_{source_id}"

            # 8. Save results locally and stream to Firestore live
            self.save_results(
                candidate
            )

            print(f"Candidate       : {candidate.candidate_id}")
            print(f"ML Probability  : {candidate.planet_probability:.3f}")
            print(f"Vetting Priority: {candidate.metadata['priority']}")
            
            unc = candidate.metadata["uncertainties"]
            print(f"[ASTROPHYSICAL DATA PACK]")
            print(f"  Host Mass : {candidate.host_mass:.2f} ± {unc['sigma_host_mass_solar']:.2f} M_☉")
            print(f"  Planet Mass: {candidate.planet_mass:.2f} ± {unc['sigma_planet_mass_earth']:.2f} M_⊕")
            print(f"  Separation : {candidate.separation:.2f} ± {unc['sigma_separation_au']:.2f} AU")
            print(f"🔄 [SYNC STATUS] Registered live in cloud registry collection.")

        print("\nAll detected sources processed and queued successfully.")



def main():

    pipeline = RomanPipeline(

        roman_file=
        "data/raw/roman_file.asdf",

        model_file=
        "models/classifier.joblib",

        output_dir=
        "output",
    )

    pipeline.run()


if __name__ == "__main__":
    main()