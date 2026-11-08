"""
firebase_sync.py

Handles authentication and secure synchronization of exoplanet candidate data
from the local Roman data pipeline up to Cloud Firestore.
"""

from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseSyncManager:

    def __init__(self, credentials_path: str = "config/serviceAccountKey.json"):
        """
        Initializes the Firebase Admin SDK connection using a service account key.
        """
        self.cred_path = Path(credentials_path)
        self.db = None

        if not self.cred_path.exists():
            print(f"[FIREBASE WARNING] Credentials missing at {self.cred_path.absolute()}")
            print("Pipeline will proceed with local-only storage fallback.")
            return

        try:
            # Prevent re-initialization exceptions if the app runs repeatedly in a notebook/session
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(self.cred_path))
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            print("Successfully authenticated and connected to Cloud Firestore.")
        except Exception as e:
            print(f"[FIREBASE ERROR] Connection failed: {e}")

    def sync_report(self, report_data: dict) -> bool:
        """
        Pushes a built AnomalyReport dictionary straight to the 'candidates' Firestore collection.
        Uses candidate_id as the explicit document path to maintain reference mapping.
        """
        if self.db is None:
            return False

        candidate_id = report_data.get("candidate_id", "UNKNOWN_CANDIDATE")
        
        try:
            # Set document structure using the clean candidate ID
            doc_ref = self.db.collection("candidates").document(candidate_id)
            doc_ref.set(report_data)
            print(f"🔄 [FIREBASE SYNC] Document '{candidate_id}' successfully synchronized to the cloud.")
            return True
        except Exception as e:
            print(f"❌ [FIREBASE SYNC ERROR] Failed to push document '{candidate_id}': {e}")
            return False
