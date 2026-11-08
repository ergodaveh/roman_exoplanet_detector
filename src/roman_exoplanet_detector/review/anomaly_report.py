"""
anomaly_report.py

Generate candidate review packages with explicit 1-sigma uncertainty tracking.
"""

from datetime import datetime
from pathlib import Path
import json


class AnomalyReport:

    def __init__(self, candidate):
        self.candidate = candidate

    def build(self):
        # Extract uncertainty fields safely from candidate metadata
        unc = getattr(self.candidate, "metadata", {}).get("uncertainties", {})
        
        return {
            "generated": 
                datetime.utcnow().isoformat(),
            "candidate_id": 
                getattr(self.candidate, "candidate_id", "UNKNOWN"),
            "status": 
                getattr(self.candidate, "current_status", "NEW"),
            
            # Core Astrophysical Values
            "planet_mass_earth": 
                getattr(self.candidate, "planet_mass", 0.0),
            "sigma_planet_mass_earth": 
                unc.get("sigma_planet_mass_earth", 0.0),
                
            "host_mass_solar": 
                getattr(self.candidate, "host_mass", 0.0),
            "sigma_host_mass_solar": 
                unc.get("sigma_host_mass_solar", 0.0),
                
            "separation_au": 
                getattr(self.candidate, "separation", 0.0),
            "sigma_separation_au": 
                unc.get("sigma_separation_au", 0.0),

            # Dimensionless Light Curve Parameters
            "mass_ratio_q": 
                unc.get("raw_fit_sigma_q", 0.0),
            "dimensionless_s": 
                unc.get("raw_fit_sigma_s", 0.0),

            # Pipeline Statistics
            "confidence": 
                getattr(self.candidate, "confidence", 0.0),
            "ml_probability": 
                getattr(self.candidate, "planet_probability", 0.0),
            "anomaly_score": 
                getattr(self.candidate, "anomaly_score", 0.0),
            "review_count": 
                self.candidate.review_count() if hasattr(self.candidate, "review_count") else 0,
            "reviews": 
                getattr(self.candidate, "reviews", []),
            "metadata": 
                getattr(self.candidate, "metadata", {})
        }

    def save_json(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        candidate_id = getattr(self.candidate, "candidate_id", "unknown")
        report_path = Path(output_dir) / f"{candidate_id}.json"

        with open(report_path, "w") as f:
            json.dump(self.build(), f, indent=4)
        return report_path

    def save_markdown(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        candidate_id = getattr(self.candidate, "candidate_id", "unknown")
        filename = Path(output_dir) / f"{candidate_id}.md"

        data = self.build()
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Exoplanet Candidate Report: {candidate_id}\n\n")
            f.write(f"**Pipeline Status:** {data['status']}  \n")
            f.write(f"**ML Planet Probability:** {data['ml_probability']:.3f}  \n")
            f.write(f"**Vetting Priority:** {data['metadata'].get('priority', 'UNKNOWN')}  \n")
            f.write(f"**Anomaly Verification Score:** {data['anomaly_score']:.1f}\n\n")

            f.write("## Derived Physical Parameters\n")
            f.write("Calculated using 1D non-linear variance propagation via the analytical Delta Method:\n\n")
            
            # Adding 'r' forces Python to ignore backslashes, printing clean LaTeX for your web frontend
            f.write(fr"* **Host Star Mass (\(M_\odot\)):** {data['host_mass_solar']:.2f} ± {data['sigma_host_mass_solar']:.2f} \(M_\odot\)\n")
            f.write(fr"* **Planet Mass (\(M_\oplus\)):** {data['planet_mass_earth']:.2f} ± {data['sigma_planet_mass_earth']:.2f} \(M_\oplus\)\n")
            f.write(f"* **Projected Orbit Separation (AU):** {data['separation_au']:.2f} ± {data['sigma_separation_au']:.2f} AU\n\n")

            f.write("## Light Curve Modeling Dimensionless Priors\n")
            f.write(fr"* **Mass Ratio (\(q\)):** {data['metadata'].get('mass_ratio_q', 0.0):.6f}\n")
            f.write(fr"* **Einstein Radius Separation (\(s\)):** {data['metadata'].get('dimensionless_separation_s', 0.0):.4f}\n")
            f.write(fr"* **Einstein Scale Metric (\(r_E\)):** {data['metadata'].get('astrophysics', {}).get('einstein_radius_au', 0.0):.2f} AU\n\n")
                
            for review in data["reviews"]:
                    f.write(f"\n### Reviewer: {review.get('reviewer', 'Anonymous')}\n")
                    f.write(f"* **Status Verdict:** {review.get('status', 'PENDING')}\n")
                    f.write(f"* **Comments:** {review.get('comments', 'None')}\n")

        return filename
