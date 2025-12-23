import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.backend import CardiacPredictor

# --- Page Config ---
st.set_page_config(
    page_title="CardioScan Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polish ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #2C3E50;
    }
    h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #34495E;
    }
    .stMetricValue {
        font-size: 2.2rem !important;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🫀") # Placeholder for logo
with col_title:
    st.title("CardioScan Pro v2.1")
    st.markdown("**Advanced Hierarchical ECG Diagnosis System** | *Powered by AI | Enterprise Edition*")

st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Patient Registration")
patient_name = st.sidebar.text_input("Full Name", "Anonymous")
c1, c2 = st.sidebar.columns(2)
patient_age = c1.number_input("Age", 0, 120, 45)
patient_gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
patient_notes = st.sidebar.text_area("Clinical Notes", "Routine Checkup")

st.sidebar.markdown("---")
st.sidebar.header("Data Source")
input_mode = st.sidebar.radio("Select Analysis Mode", ["📈 Signal Analysis (.npy)", "👁️ Visual Analysis (Image)"])


# Initialize Predictor (Cached)
@st.cache_resource
def get_predictor():
    return CardiacPredictor()

try:
    predictor = get_predictor()
    st.sidebar.success("✅ AI Core Active")
except Exception as e:
    st.sidebar.error(f"❌ AI Core Failed: {e}")
    st.stop()


# --- LOGIC HANDLING ---

def display_medical_report(prediction, mode="Signal"):
    """
    Renders a sterile, professional Clinical Report Card (EMR Style).
    Uses Streamlit Components to isolate HTML/CSS from Markdown parsing issues.
    """
    import streamlit.components.v1 as components
    
    diag = prediction.get('diagnosis', 'Unknown')
    conf = prediction.get('confidence', 0.0)
    top3 = prediction.get('top3', {})
    triage = prediction.get('triage', 'Visual')
    
    # Clinical Color Codes
    if diag in ['NORM', 'Sinus_Rhythm']:
        theme_color = "#2e7d32" # Medical Green
        bg_color = "#e8f5e9"
        status_text = "NORMAL VARIANT"
        severity = "LOW PRIORITY"
    elif triage == "Rhythm":
        theme_color = "#ef6c00" # Clinical Amber
        bg_color = "#fff3e0"
        status_text = "RHYTHM DISTURBANCE"
        severity = "MODERATE PRIORITY"
    else:
        theme_color = "#c62828" # Clinical Red
        bg_color = "#ffebee"
        status_text = "MORPHOLOGICAL ANOMALY"
        severity = "URGENT ATTENTION"

    # Differential Diagnosis Rows
    rows_html = ""
    for condition, prob in top3.items():
        pct = prob * 100
        row_color = "#2196f3" if prob == max(top3.values()) else "#b0bec5"
        rows_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eeeeee; color: #37474f; font-weight: 500;">{condition}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eeeeee; text-align: right; color: #546e7a;">{pct:.1f}%</td>
            <td style="padding: 8px; border-bottom: 1px solid #eeeeee;">
                <div style="background-color: #eceff1; height: 6px; border-radius: 3px; width: 100%;">
                    <div style="background-color: {row_color}; height: 6px; border-radius: 3px; width: {pct}%;"></div>
                </div>
            </td>
        </tr>"""

    # Assemble Report - Full HTML Page for Iframe
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 10px;
                background-color: transparent;
            }}
            .card {{
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .header {{
                border-bottom: 2px solid {theme_color};
                padding-bottom: 10px;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .metric-box {{
                padding: 15px;
                border-radius: 4px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <div>
                    <span style="font-size: 0.85em; color: #757575; text-transform: uppercase; letter-spacing: 1px;">Clinical Diagnosis</span>
                    <h2 style="margin: 0; color: #2c3e50; font-size: 1.8em;">{diag}</h2>
                </div>
                <div style="text-align: right;">
                    <div style="background-color: {theme_color}; color: white; padding: 4px 12px; border-radius: 2px; font-size: 0.8em; font-weight: bold;">{severity}</div>
                    <div style="color: {theme_color}; font-size: 0.8em; margin-top: 4px;">{status_text}</div>
                </div>
            </div>

            <div class="grid">
                <div class="metric-box" style="background-color: #f8f9fa; border-left: 4px solid #b0bec5;">
                    <div style="font-size: 0.8em; color: #546e7a;">AI CONFIDENCE</div>
                    <div style="font-size: 1.4em; font-weight: 500; color: #263238;">{conf:.1%}</div>
                    <div style="font-size: 0.7em; color: #78909c;">Probability index</div>
                </div>
                <div class="metric-box" style="background-color: {bg_color}; border-left: 4px solid {theme_color};">
                    <div style="font-size: 0.8em; color: {theme_color}; opacity: 0.8;">CLASSIFICATION</div>
                    <div style="font-size: 1.4em; font-weight: 500; color: {theme_color};">{mode} / {triage}</div>
                    <div style="font-size: 0.7em; color: {theme_color}; opacity: 0.8;">Taxonomy Stage III</div>
                </div>
            </div>

            <h4 style="margin: 0 0 10px 0; color: #455a64; font-size: 0.9em; text-transform: uppercase;">Differential Probabilities</h4>
            <table>
                <thead>
                    <tr style="background-color: #f5f5f5; color: #616161; text-align: left;">
                        <th style="padding: 8px; border-bottom: 2px solid #eeeeee;">Condition</th>
                        <th style="padding: 8px; border-bottom: 2px solid #eeeeee; text-align: right;">Likelihood</th>
                        <th style="padding: 8px; border-bottom: 2px solid #eeeeee;">Indicator</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div style="font-size: 0.75em; color: #9e9e9e; margin-top: 15px; font-style: italic;">
                * Computer Aided Diagnosis (CADx) Estimate. Not a definitive medical confirmation. correlate with clinical history.
            </div>
        </div>
    </body>
    </html>
    """
    
    # Use components.html instead of markdown to sand-box the HTML and avoid markdown parser errors
    components.html(html_content, height=600, scrolling=True)


# --- MODE 1: SIGNAL ANALYSIS ---
if input_mode == "📈 Signal Analysis (.npy)":
    st.sidebar.subheader("Signal Source")
    source_type = st.sidebar.radio("Source", ["Upload File", "Load Random Patient"])
    
    signal_data = None
    ground_truth = None
    
    if source_type == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload .npy file", type=['npy'])
        if uploaded_file:
            signal_data = np.load(uploaded_file)
            
    else: # Random Patient
        if st.sidebar.button("🎲 Fetch Random Sample"):
            with st.spinner("Retrieving patient record..."):
                DATA_ROOT = "../production/data"
                # Fallback paths for Docker vs Local
                if not os.path.exists(DATA_ROOT): DATA_ROOT = "production/data" # Local run
                if not os.path.exists(DATA_ROOT): DATA_ROOT = "data" # Docker mount
                
                CSV_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
                
                if os.path.exists(CSV_PATH):
                    df = pd.read_csv(CSV_PATH)
                    # Filter for interesting cases
                    mask = ~df['label'].isin(['NORM', 'Sinus_Rhythm'])
                    if 'diagnostic_superclass' in df.columns:
                         mask = mask & ~df['diagnostic_superclass'].isin(['NORM', 'Sinus_Rhythm'])
                    
                    sample = df[mask].sample(1).iloc[0]
                    fname = sample.get('filename_hr')
                    if pd.isna(fname): fname = sample.get('filename')
                    
                    # Clean filename
                    fname = str(fname).replace('.png', '').replace('.npy', '')
                    
                    # Try finding file
                    # In this strict env, we might not have all 20k files. 
                    # Simulating success if file missing for UI demo? No, let's try real.
                    fpath = os.path.join(DATA_ROOT, fname + ".npy")
                    
                    if os.path.exists(fpath):
                        signal_data = np.load(fpath)
                        ground_truth = sample['label']
                        st.session_state['signal_data'] = signal_data
                        st.session_state['ground_truth'] = ground_truth
                    else:
                        st.sidebar.error(f"Sample file {fname}.npy not found in local subset.")
                else:
                    st.sidebar.error(f"Database not found at {CSV_PATH}")

        # Restore state
        if 'signal_data' in st.session_state:
            signal_data = st.session_state['signal_data']
            ground_truth = st.session_state.get('ground_truth', 'Unknown')
            
            # --- PERSISTENCE FOR PERFORMANCE ---
            # Generate a "Data ID" to detect changes in signal or patient info
            data_id = hash((signal_data.tobytes(), patient_name, patient_age, patient_gender))
            if st.session_state.get('last_signal_id') != data_id:
                # Clear stale results if data or metadata changed
                st.session_state['last_signal_id'] = data_id
                st.session_state['engine_b_report'] = None
                st.session_state['engine_b_prediction'] = None

    # DISPLAY SIGNAL
    if signal_data is not None:
        st.divider()
        st.subheader("High-Fidelity 12-Lead Report")

        from utils.plotting import plot_12_lead_ecg, plot_interactive_3d
        
        # 1. 3D Visualizer (Top)
        with st.expander("✨ 3D Holographic View (Interactive)", expanded=True):
            # Layout: Controls | Graph
            c_graph, c_ctrl = st.columns([4, 1])
            
            with c_ctrl:
                st.markdown("###### 🛠 Controls")
                theme = st.selectbox("Color Theme", ["Clinical Groups", "Monochrome Cyan", "Retro Neon"], index=0, help="Group leads by anatomy or use unified colors.")
                line_width = st.slider("Thickness", 1.0, 5.0, 2.5, 0.5, help="Adjust signal trace width.")
                opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.1)
                st.caption("💡 Click Legend items to toggle leads.")
            
            with c_graph:
                fig_3d = plot_interactive_3d(signal_data, theme=theme, line_width=line_width, opacity=opacity)
                st.plotly_chart(fig_3d, use_container_width=True)
            
        # 2. Static Report
        
        # Prepare Metadata
        pf_name = "Uploaded_File"
        if source_type == "Upload File" and uploaded_file:
            pf_name = uploaded_file.name
        elif source_type == "Load Random Patient":
            pf_name = "Database_Sample_ID"
            
        patient_meta = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "notes": patient_notes
        }

        # --- CACHED REPORT GENERATION ---
        if st.session_state.get('engine_b_report') is None:
            with st.spinner("Generating Medical-Grade Trace..."):
                report_path = plot_12_lead_ecg(
                    signal_data, 
                    patient_meta=patient_meta,
                    original_filename=pf_name,
                    save_dir="data/engine_b" 
                )
                st.session_state['engine_b_report'] = report_path
        
        report_path = st.session_state['engine_b_report']
        
        # Display
        st.image(report_path, caption=f"Report Generated for {patient_name}", use_container_width=True)
        
        # Download Button
        with open(report_path, "rb") as file:
            st.download_button(
                label="📥 Download Clinical Report (PNG)",
                data=file,
                file_name=os.path.basename(report_path),
                mime="image/png"
            )
        
        st.success(f"✅ Report saved to: {report_path}")
        
        # --- CACHED PREDICTION ---
        if st.session_state.get('engine_b_prediction') is None:
            with st.spinner("Processing Signal through Neural Hierarchy..."):
                result = predictor.predict(signal_data)
                st.session_state['engine_b_prediction'] = result
            
        display_medical_report(st.session_state['engine_b_prediction'], mode="Signal")
        
        if ground_truth:
            st.caption(f"Medical Record Label: {ground_truth}")

    else:
        st.info("👈 Please load patient data from the sidebar to begin analysis.")


# --- MODE 2: VISUAL ANALYSIS ---
elif input_mode == "👁️ Visual Analysis (Image)":
    st.sidebar.subheader("Paper Scan Input")
    img_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if img_file:
        from PIL import Image
        image = Image.open(img_file)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image, caption="Scanned Document", use_container_width=True)
            
        with c2:
            st.write("### AI Vision Engine")
            
            # Lazy Load Visual Predictor
            @st.cache_resource
            def get_visual_predictor():
                from utils.visual_backend import VisualPredictor
                return VisualPredictor()
            
            try:
                vis_predictor = get_visual_predictor()
                
                # Check if we should clear vision cache
                vis_id = hash((img_file.name, patient_name, patient_age, patient_gender))
                if vis_id != st.session_state.get('last_vis_id'):
                    st.session_state['last_vis_id'] = vis_id
                    st.session_state['engine_a_result'] = None

                if st.button("Analyze Scan") or st.session_state.get('engine_a_result') is not None:
                    if st.session_state.get('engine_a_result') is None:
                        with st.spinner("Extracting features & Classifying..."):
                            # Gather Patient Info from Sidebar
                            patient_meta = {
                                "name": patient_name,
                                "age": patient_age,
                                "gender": patient_gender,
                                "notes": patient_notes
                            }
                            
                            vis_result = vis_predictor.predict(image, patient_metadata=patient_meta)
                            st.session_state['engine_a_result'] = vis_result
                        
                    vis_result = st.session_state['engine_a_result']
                        
                    if "saved_files" in vis_result and vis_result["saved_files"]:
                         st.success(f"📁 Image Data Saved: {os.path.basename(vis_result['saved_files']['pre'])}")
                        
                    if "error" in vis_result:
                        st.error(f"Analysis Failed: {vis_result['error']}")
                    else:
                        display_medical_report(vis_result, mode="Visual")
                        
            except Exception as e:
                st.error(f"Visual Engine Error: {e}")
    else:
        st.info("👈 Please upload an ECG image from the sidebar.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7F8C8D; font-size: 0.8em;'>
        CardioScan Pro v2.1 | Confidential Medical Device Software | © 2025 PBLS Team
    </div>
    """, 
    unsafe_allow_html=True
)
