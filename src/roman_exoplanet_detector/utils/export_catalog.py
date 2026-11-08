"""
export_catalog.py

Downloads live candidate records from Cloud Firestore and exports them into 
a clean, publication-ready Excel spreadsheet catalog for scientific distribution.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone


def run_catalog_export(credentials_path: str = "config/serviceAccountKey.json", output_path: str = "output/exports/"):
    """
    Fetches the live 'candidates' collection from the cloud database, 
    unpacks nested metrics, and writes out a clean, multi-field Excel sheet.
    """
    cred_file = Path(credentials_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cred_file.exists():
        print(f"❌ [EXPORT ERROR] Service account key missing at: {cred_file.absolute()}")
        return False

    print("🛰️  [CONNECTING] Querying Cloud Firestore registry database...")
    
    # Initialize the core client connection
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_file))
        firebase_admin.initialize_app(cred)
        
    db = firestore.client()
    
    # Stream all live documents out of your cloud collection
    docs = db.collection("candidates").stream()
    raw_records = [doc.to_dict() for doc in docs]
    
    if not raw_records:
        print("⚠️  [EXPORT WARNING] No processed candidates discovered in the cloud database collection.")
        return False

    print(f"📥 [DOWNLOADING] Retrieved {len(raw_records)} live candidate data records from the cloud. Parsing profiles...")

    processed_rows = []
    
    # Flatten the multi-tiered JSON dictionary parameters into a row-column tabular grid structure
    for record in raw_records:
        meta = record.get("metadata", {})
        unc = meta.get("uncertainties", {})
        diag = meta.get("chromatic_diagnostics", {})
        
        row = {
            "CANDIDATE_ID": record.get("candidate_id", "UNKNOWN"),
            "VETTING_STATUS": record.get("status", "NEW"),
            "PRIORITY": meta.get("priority", "UNKNOWN"),
            "ML_PROBABILITY": record.get("ml_probability", 0.0),
            "ANOMALY_SCORE": record.get("anomaly_score", 0.0),
            
            # Core Astrophysical Quantities & Propagated 1-Sigma Variances
            "HOST_MASS_M_SUN": record.get("host_mass_solar", 0.0),
            "SIGMA_HOST_MASS": unc.get("sigma_host_mass_solar", 0.0),
            
            "PLANET_MASS_M_EARTH": record.get("planet_mass_earth", 0.0),
            "SIGMA_PLANET_MASS": unc.get("sigma_planet_mass_earth", 0.0),
            
            "SEPARATION_AU": record.get("separation_au", 0.0),
            "SIGMA_SEPARATION": unc.get("sigma_separation_au", 0.0),
            
            # Dimensionless Modelling Priors
            "MASS_RATIO_Q": meta.get("mass_ratio_q", 0.0),
            "SIGMA_Q": unc.get("raw_fit_sigma_q", 0.0),
            "SEPARATION_S": meta.get("dimensionless_separation_s", 0.0),
            "SIGMA_S": unc.get("raw_fit_sigma_s", 0.0),
            
            # Chromatic Diagnostics
            "STELLAR_CLASSIFICATION": diag.get("classified_stellar_profile", "Unknown"),
            "COLOR_INDEX_F087_F184": diag.get("color_index_087_184", 0.0),
            
            "REVIEW_COUNT": record.get("review_count", 0),
            "TIMESTAMP_UTC": record.get("generated", datetime.now(timezone.utc).isoformat())
        }
        processed_rows.append(row)

    # Convert to standard Pandas DataFrame data structure
    df = pd.DataFrame(processed_rows)
    
    # Chronologically sort rows by ID for clean spreadsheet presentation
    df = df.sort_values(by="CANDIDATE_ID").reset_index(drop=True)

    # 4. Write out the binary data layers using openpyxl engine
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = output_dir / f"roman_exoplanet_catalog_{datestamp}.xlsx"
    
    # Use pandas ExcelWriter to construct elegant formatting traits
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Roman Exoplanet Census", index=False)
        
        # Grab the openpyxl workbook worksheet object to auto-adjust column widths for scannability
        workbook = writer.book
        worksheet = writer.sheets["Roman Exoplanet Census"]
        
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    print(f"📊 [SUCCESS] Production Excel Catalog generated and saved to: {excel_filename.absolute()}")
    return True


if __name__ == "__main__":
    run_catalog_export()
