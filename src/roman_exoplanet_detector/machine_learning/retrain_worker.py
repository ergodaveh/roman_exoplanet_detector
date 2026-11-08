"""
retrain_worker.py

Asynchronous background worker that harvests vetted exoplanet classifications 
from Cloud Firestore, runs multi-threaded estimator retraining, and atomic hot-swaps 
the local machine learning classifier binaries.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Align paths to ensure internal modules load cleanly
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from roman_exoplanet_detector.machine_learning.model_training import CandidateClassifier
from roman_exoplanet_detector.machine_learning.feature_builder import FeatureBuilder
from roman_exoplanet_detector.microlensing.event_detection import EventDetector
from roman_exoplanet_detector.microlensing.paczynski import paczynski


def build_synthetic_augmentation_data(n_samples=60):
    """
    Generates balanced fallback datasets using our known physical engines 
    to prevent class imbalance convergence failure during cold starts.
    """
    rows = []
    time = np.linspace(0.0, 60.0, 100)
    event_detector = EventDetector()

    for idx in range(n_samples):
        has_planet = (idx % 2 == 0)
        
        t0, tE = 30.0, 22.5
        u0 = 0.15 if has_planet else 0.45
        blending_fraction_fb = 0.75

        u = np.sqrt(u0**2 + ((time - t0) / tE)**2)
        magnification_A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

        if has_planet:
            planetary_anomaly = 0.18 * np.exp(-((time - 34.2) ** 2) / (2 * (0.4 ** 2)))
            total_mag = magnification_A + planetary_anomaly
            label = 1
        else:
            total_mag = magnification_A
            label = 0

        flux = 1.0 * (blending_fraction_fb * total_mag + (1.0 - blending_fraction_fb))
        error = 0.008 * np.sqrt(flux)
        flux += np.random.normal(0.0, error)

        fit_res = event_detector.fit_event(time, flux)
        model_flux = paczynski(time, *fit_res["params"])
        
        fb = FeatureBuilder(time=time, flux=flux, flux_error=error, model_flux=model_flux)
        feat_df = fb.feature_dataframe()
        
        if not feat_df.empty:
            # Flatten to 1D series dictionary mapping row primitives safely
            feat_dict = feat_df.iloc[0].to_dict()
            feat_dict["label"] = label
            rows.append(feat_dict)
            
    return rows


def execute_async_retrain():
    print(f"\n🔄 [ML WORKER START] Initialization triggered at {datetime.now(timezone.utc).isoformat()}")
    
    script_dir = Path(__file__).resolve().parent  # points to machine_learning/
    repo_root = script_dir.parents[1]            # goes up two levels to roman_exoplanet_detector/
    
    cred_path = Path(r"C:\Firebase-app\roman-exoplanet-detector\roman_exoplanet_detector\config\serviceAccountKey.json")
    model_target_path = Path(r"C:\Firebase-app\roman-exoplanet-detector\models\classifier.joblib")

    
    if not cred_path.exists():
        print(f"❌ [WORKER ABORT] Configuration key missing at: {cred_path.absolute()}")
        return

    # 1. Connect securely to Cloud Firestore
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("📡 Harvesting expert vetted records from Firestore...")
    
    # 2. Extract only expert manual verifications using modern FieldFilter objects
    candidates_ref = db.collection("candidates")
    filter_status = FieldFilter("status", "in", ["VERIFIED_PLANET", "FALSE_POSITIVE", "STELLAR_FLARE_REJECT"])
    vetted_stream = candidates_ref.where(filter=filter_status).stream()
    
    cloud_rows = []
    for doc in vetted_stream:
        data = doc.to_dict()
        status = data.get("status")
        meta = data.get("metadata", {})
        
        t_arr = meta.get("raw_time", [])
        f_arr = meta.get("raw_flux", [])
        e_arr = meta.get("raw_err", [])
        
        if t_arr and f_arr:
            event_detector = EventDetector()
            fit_res = event_detector.fit_event(np.array(t_arr), np.array(f_arr))
            model_flux = paczynski(np.array(t_arr), *fit_res["params"])
            
            fb = FeatureBuilder(time=np.array(t_arr), flux=np.array(f_arr), flux_error=np.array(e_arr), model_flux=model_flux)
            feat_df = fb.feature_dataframe()
            
            if not feat_df.empty:
                feat_dict = feat_df.iloc[0].to_dict()
                feat_dict["label"] = 1 if status == "VERIFIED_PLANET" else 0
                cloud_rows.append(feat_dict)

    print(f"📥 Successfully harvested {len(cloud_rows)} expert vetted profiles from the cloud.")

    # 3. Infuse synthetic data vectors to protect the classifier against cold-start matrix failures
    print("Infusing physical augmentation matrices to balance classification parameters...")
    synthetic_rows = build_synthetic_augmentation_data(n_samples=60)
    
    comprehensive_dataset = synthetic_rows + cloud_rows
    training_df = pd.DataFrame(comprehensive_dataset)

    # 4. Split variables into features matrix X and target labels y
    X = training_df.drop(columns=["label"])
    y = training_df["label"]

    print(f"🏋️  Retraining estimator ensemble on comprehensive array matrix shape: {X.shape}")
    
    # 5. Initialize your real research-grade model structure and run .train() directly
    classifier = CandidateClassifier()
    classifier.train(X, y)

    # 6. Execute an atomic file-system hot-swap
    temp_model_path = Path("models/classifier_new.joblib")
    model_target_path.parent.mkdir(parents=True, exist_ok=True)
    
    classifier.save(temp_model_path)
    os.replace(temp_model_path, model_target_path)
    
    print(f"🚀 [SUCCESS] Estimator hot-swap finalized. Binarized matrix deployed to: {model_target_path.absolute()}\n")


if __name__ == "__main__":
    execute_async_retrain()
