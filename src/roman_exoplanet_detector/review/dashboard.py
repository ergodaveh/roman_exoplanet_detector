"""
dashboard.py

Streamlit web application for Roman exoplanet verification featuring secure 
Firebase User Authentication gating for manual peer-review writes.
"""

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timezone

# -------------------------------------------------------------
# SECURE FIREBASE CONFIGURATION WINDOW
# -------------------------------------------------------------
# Paste your newly generated app registration apiKey string right here:
FIREBASE_WEB_API_KEY = "AIzaSyD_P7ekSqeDjJyCBij2CxsgM_lOV-Nu8Vc"
# -------------------------------------------------------------

st.set_page_config(
    page_title="Roman Exoplanet Review Portal",
    page_icon="🔭",
    layout="wide"
)

def calculate_theoretical_signal(t_arr, q, t_planet, width):
    t = np.array(t_arr)
    return q * np.exp(-0.5 * ((t - t_planet) / width) ** 2)

@st.cache_resource
def get_firestore_client():
    if not firebase_admin._apps:
        try:
            # Check if running on the live Streamlit Cloud server with secrets injected
            if hasattr(st, "secrets") and "firebase" in st.secrets:
                # Convert the encrypted Streamlit secrets mapping securely into a standard dict buffer
                firebase_secrets = dict(st.secrets["firebase"])
                
                # Double-check that inner keys like private_key aren't empty strings
                if firebase_secrets.get("private_key"):
                    cred = credentials.Certificate(firebase_secrets)
                else:
                    raise KeyError("Firebase private_key missing or unparsed inside the secrets vault.")
            else:
                # Fall back to your absolute local file path when running via run_pipeline.bat locally
                cred_path = "config/serviceAccountKey.json"
                cred = credentials.Certificate(cred_path)
                
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Failed to load cloud configuration: {e}")
            return None
    return firestore.client()


db = get_firestore_client()

# Initialize session state variables to track user logins across re-runs
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

st.title("🔭 Nancy Grace Roman Space Telescope")
st.subheader("Exoplanet Candidate Real-Time Verification Registry")

if db is None:
    st.warning("Running dashboard in sandbox mode. Connect serviceAccountKey.json to stream cloud records.")
else:
    docs = db.collection("candidates").stream()
    rows = [doc.to_dict() for doc in docs]
        
    if not rows:
        st.info("No candidates processed in the cloud database registry yet.")
    else:
        df = pd.DataFrame(rows)
        
        # --- SIDEBAR INTERFACE TOOLS ---
        st.sidebar.header("Survey Filtration System")
        if "planet_mass_earth" in df.columns and not df["planet_mass_earth"].empty:
            mass_min = float(df["planet_mass_earth"].min())
            mass_max = float(df["planet_mass_earth"].max())
            if mass_min == mass_max:
                mass_min = 0.0
                mass_max = max(100.0, mass_max * 1.2)
        else:
            mass_min, mass_max = 0.0, 5000.0

        selected_mass = st.sidebar.slider("Max Planet Mass (M_⊕)", mass_min, mass_max, mass_max)
        filtered_df = df[df["planet_mass_earth"] <= selected_mass]
        
        st.sidebar.markdown("---")
        st.sidebar.header("Workspace Diagnostics")
        show_best_fit = st.sidebar.checkbox("Overlay Best-Fit Model Trace", value=False)
        
        # -------------------------------------------------------------
        # NATIVE CLIENT-SIDE USER AUTHENTICATION SIDEBAR PANEL
        # -------------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.header("🔒 Astronomer Access Gate")
        
        if st.session_state.auth_user is None:
            st.sidebar.caption("Sign in with your consortium credentials to unlock vetting privileges.")
            with st.sidebar.form(key="login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                login_submit = st.form_submit_button("Authenticate Token")
                
                if login_submit:
                     
                    base_endpoint = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
                    url = f"{base_endpoint}?key={FIREBASE_WEB_API_KEY.strip()}"
                    
                    
                    payload = {
                        "email": email.strip(),
                        "password": password.strip(),
                        "returnSecureToken": True
                    }
                    
                    headers = {"Content-Type": "application/json"}
                    
                    try:
                        response = requests.post(url, json=payload, headers=headers)
                        
                        if response.text:
                            res_data = response.json()
                            
                            if response.status_code == 200:
                                st.session_state.auth_user = {
                                    "email": res_data["email"],
                                    "localId": res_data["localId"],
                                    "idToken": res_data["idToken"]
                                }
                                st.sidebar.success(f"Welcome, {res_data['email']}!")
                                st.rerun()
                            else:
                                error_msg = res_data.get("error", {}).get("message", "INVALID_CREDENTIALS")
                                st.sidebar.error(f"Authentication Failed: {error_msg}")
                        else:
                            st.sidebar.error(f"Empty server response. Status code: {response.status_code}")
                            
                    except requests.exceptions.JSONDecodeError:
                        st.sidebar.error("Server returned an invalid non-JSON payload response.")
                        print(f"❌ [AUTH DEBUG ROW] Raw Server Content: {response.text}")
                    except Exception as e:
                        st.sidebar.error(f"Network error encountered: {e}")


        else:
            st.sidebar.success(f"Logged in as: `{st.session_state.auth_user['email']}`")
            if st.sidebar.button("Sign Out / Terminate Session"):
                st.session_state.auth_user = None
                st.rerun()

        # Excel Download Tool Section
        st.sidebar.markdown("---")
        st.sidebar.header("Data Distribution")
        
        @st.cache_data(ttl=60)
        def generate_catalog_buffer():
            import io
            buffer = io.BytesIO()
            docs_export = db.collection("candidates").stream()
            raw_records = [d.to_dict() for d in docs_export]
            if not raw_records: return None
            processed_rows = []
            for record in raw_records:
                meta = record.get("metadata", {})
                unc = meta.get("uncertainties", {})
                diag = meta.get("chromatic_diagnostics", {})
                processed_rows.append({
                    "CANDIDATE_ID": record.get("candidate_id", "UNKNOWN"),
                    "VETTING_STATUS": record.get("status", "NEW"),
                    "PLANET_MASS_M_EARTH": record.get("planet_mass_earth", 0.0),
                    "SIGMA_PLANET_MASS": unc.get("sigma_planet_mass_earth", 0.0),
                    "SEPARATION_AU": record.get("separation_au", 0.0),
                    "STELLAR_CLASSIFICATION": diag.get("classified_stellar_profile", "Unknown")
                })
            df_export = pd.DataFrame(processed_rows).sort_values(by="CANDIDATE_ID").reset_index(drop=True)
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, sheet_name="Roman Exoplanet Census", index=False)
            return buffer.getvalue()

        excel_data = generate_catalog_buffer()
        if excel_data:
            st.sidebar.download_button(
                label="📥 Download Excel Catalog", data=excel_data,
                file_name=f"roman_exoplanet_catalog_{datetime.now(timezone.utc).strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Metrics Display Grid
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flagged Alerts", len(df))
        col2.metric("Filtered Candidates", len(filtered_df))
        col3.metric("Max Probability Found", f"{df['ml_probability'].max():.3f}" if "ml_probability" in df.columns else "N/A")
        col4.metric("Mean Host Star Mass", f"{df['host_mass_solar'].mean():.2f} M_☉" if "host_mass_solar" in df.columns else "N/A")
        
        st.markdown("---")
        
        # Registry Table Grid Layout
        st.write("### Target Cadence Summary Registry")
        display_df = pd.DataFrame()
        display_df["ID"] = filtered_df["candidate_id"]
        display_df["Vetting Status"] = filtered_df["status"]
        display_df["ML Prob."] = filtered_df["ml_probability"].map(lambda x: f"{x:.3f}")
        display_df["Host Star Mass"] = filtered_df.apply(lambda r: f"{r['host_mass_solar']:.2f} ± {r['sigma_host_mass_solar']:.2f} M_☉", axis=1)
        display_df["Planet Mass"] = filtered_df.apply(lambda r: f"{r['planet_mass_earth']:.2f} ± {r['sigma_planet_mass_earth']:.2f} M_⊕", axis=1)
        display_df["Proj. Orbit Separation"] = filtered_df.apply(lambda r: f"{r['separation_au']:.2f} ± {r['sigma_separation_au']:.2f} AU", axis=1)
        display_df["Anomaly Score"] = filtered_df["anomaly_score"]
        
        selection = st.dataframe(
            display_df, width="stretch", hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )

        active_target_id = None
        if selection and selection.get("selection", {}).get("rows"):
            selected_row_idx = selection["selection"]["rows"][0] # Extract integer position from selected rows list
            active_target_id = display_df.iloc[selected_row_idx]["ID"]
        else:
            # --- SOLID ABSOLUTE PRIMITIVE STRING FALLBACK ---
            # Isolate row position index 0 of the string column explicitly 
            active_target_id = display_df["ID"].iloc[0] if not display_df.empty else None
            
        st.markdown("### 📊 Chromatic Structural Analysis")
        tab1, tab2 = st.tabs(["Stellar Field Color-Magnitude Diagram (CMD)", "Photometry Light Curve Inspection"])
        
        with tab1:
            st.write("#### Field Photometry CMD Profile")
            cmd_rows = []
            for item in rows:
                diag = item.get("metadata", {}).get("chromatic_diagnostics", {})
                if diag:
                    cmd_rows.append({
                        "id": item.get("candidate_id"),
                        "color_index": diag.get("color_index_087_184", 0.0),
                        "mag_w146": -2.5 * np.log10(np.median(item.get("metadata", {}).get("raw_flux", [100.0]))),
                        "profile": diag.get("classified_stellar_profile", "Unknown")
                    })
            
            if cmd_rows:
                cmd_df = pd.DataFrame(cmd_rows)
                fig_cmd = go.Figure()
                for profile_type in cmd_df["profile"].unique():
                    sub_df = cmd_df[cmd_df["profile"] == profile_type]
                    fig_cmd.add_trace(go.Scatter(
                        x=sub_df["color_index"], y=sub_df["mag_w146"], mode='markers+text',
                        name=profile_type, text=sub_df["id"], textposition="top center",
                        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey'))
                    ))
                fig_cmd.update_layout(
                    xaxis_title="Color Index (F087 - F184 Ratio)", yaxis_title="Apparent Magnitude (W146)",
                    template="plotly_white", height=450, yaxis=dict(autorange="reversed", showgrid=True, gridcolor='rgba(235, 235, 235, 1)')
                )
                st.plotly_chart(fig_cmd, width='stretch', key="cmd_stellar_chart")
            else:
                st.info("No chromatic metadata payloads detected.")

        with tab2:
            if active_target_id:
                st.info(f"Viewing active target telemetry matrix profile context: **{active_target_id}**")
                target_data = next((item for item in rows if item.get("candidate_id") == active_target_id), {})
                meta_obj = target_data.get("metadata", {})
                
                t_arr = meta_obj.get("raw_time", [])
                f_arr = meta_obj.get("raw_flux", [])
                e_arr = meta_obj.get("raw_err", [])
                
                if t_arr and f_arr:
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(
                        x=t_arr, y=f_arr, mode='markers', name='Observed Photometry Data',
                        marker=dict(color='rgba(31, 119, 180, 0.6)', size=4),
                        error_y=dict(type='data', array=e_arr, visible=True, color='rgba(31, 119, 180, 0.15)', thickness=1)
                    ))
                    
                    if show_best_fit:
                        fit_curve = np.array(meta_obj.get("fitted_model_flux", f_arr))
                        fig_lc.add_trace(go.Scatter(
                            x=t_arr, y=fit_curve, mode='lines', name='Model Best-Fit Line', line=dict(color='#ff7f0e', width=2.5)
                        ))
                    
                    fig_lc.update_layout(
                        title=f"Roman WFI Flux Photometry Profile — Event Reference: {active_target_id}",
                        xaxis_title="Time (BJD)", yaxis_title="Normalized Relative Flux Level", template="plotly_white", height=450
                    )
                    st.plotly_chart(fig_lc, width='stretch', key="light_curve_photometry_chart")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("#### Telemetry Metadata Payload")
                    st.json(meta_obj)
                
                # --- PROTECTION LOCK INTEGRATED IN WORKSPACE COLUMN ---
                with c2:
                    st.write("#### 📝 Peer-Review Verification Workspace")
                    review_list = target_data.get("reviews", [])
                    st.write(f"Logged Review Packages: `{len(review_list)}`")
                    
                    if review_list:
                        for rev in review_list:
                            with st.container(border=True):
                                st.markdown(f"**Reviewer:** {rev.get('reviewer')} | *{rev.get('status')}*")
                                st.write(f"*{rev.get('comments')}*")
                    else:
                        st.info("No manual peer-reviews finalized for this target system yet.")
                    
                    st.markdown("---")
                    
                    # Vetting gate conditional check block
                    if st.session_state.auth_user is None:
                        st.warning("🔒 **Review Workspace Locked.** You must authenticate your user account in the left sidebar to unlock entry options.")
                    else:
                        st.success(f"🔓 **Workspace Unlocked.** Authorized User: `{st.session_state.auth_user['email']}`")
                        with st.form(key=f"vetting_form_{active_target_id}", clear_on_submit=True):
                            reviewer_name = st.text_input("Astronomer Name / Consortium ID", value=st.session_state.auth_user['email'], disabled=True)
                            vetting_status = st.selectbox("Verification Verdict", options=["NEW", "UNDER_REVIEW", "VERIFIED_PLANET", "FALSE_POSITIVE", "STELLAR_FLARE_REJECT"])
                            review_comments = st.text_area("Scientific Analysis Comments")
                            submit_button = st.form_submit_button("Finalize & Commit Review to Cloud")
                            
                            if submit_button:
                                if not review_comments.strip():
                                    st.error("Scientific comments cannot be left blank for verification records.")
                                else:
                                    new_review_payload = {
                                        "reviewer": reviewer_name,
                                        "status": vetting_status,
                                        "comments": review_comments.strip(),
                                        "timestamp": datetime.now(timezone.utc).isoformat()
                                    }
                                    try:
                                        doc_ref = db.collection("candidates").document(active_target_id)
                                        doc_ref.update({
                                            "reviews": firestore.ArrayUnion([new_review_payload]),
                                            "status": vetting_status
                                        })
                                        st.success(f"🔄 Document '{active_target_id}' updated in the cloud!")
                                        st.toast("Peer commentary pushed live!", icon="🚀")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Database Transaction Failed: {e}")
