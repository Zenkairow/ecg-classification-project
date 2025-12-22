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
    st.title("CardioScan Pro")
    st.markdown("**Advanced Hierarchical ECG Diagnosis System** | *Powered by AI*")

st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Patient Data Input")

input_mode = st.sidebar.radio("Select Analysis Mode", ["📈 Signal Analysis (.npy)", "👁️ Visual Analysis (Image)"])

st.sidebar.markdown("---")

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
    """Reusable function to display professional results"""
    
    diag = prediction.get('diagnosis', 'Unknown')
    conf = prediction.get('confidence', 0.0)
    top3 = prediction.get('top3', {})
    triage = prediction.get('triage', 'Visual')
    
    # Semantic Color Logic
    if diag in ['NORM', 'Sinus_Rhythm']:
        status_color = "normal" # Streamlit green
        box_color = "green"
        status_text = "NORMAL / LOW RISK"
    elif triage == "Rhythm":
        status_color = "off" # Streamlit gray/neutral
        box_color = "orange"
        status_text = "ARRHYTHMIA DETECTED"
    else:
        status_color = "inverse" # Streamlit red
        box_color = "red"
        status_text = "STRUCTURAL / MORPHOLOGICAL ABNORMALITY"

    # 1. Top Level Status
    st.markdown(f"### Diagnostic Report: {mode}")
    
    # Metrics Row
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label="Primary Diagnosis", value=diag, delta=None)
    with m2:
        st.metric(label="Confidence Score", value=f"{conf:.1%}", delta=status_text, delta_color=status_color)
    with m3:
        st.metric(label="Triage Category", value=triage)

    # 2. Detailed Visualization
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.subheader("Probability Distribution")
        # Custom Chart
        chart_data = pd.DataFrame(
            {"Condition": list(top3.keys()), "Probability": list(top3.values())}
        ).sort_values("Probability", ascending=True)
        
        st.bar_chart(chart_data, x="Condition", y="Probability", use_container_width=True, color="#3498DB")
        
    with c2:
        st.subheader("Clinical interpretation")
        if box_color == "green":
             st.success(f"**{diag}**: This ECG pattern suggests normal cardiac function. {status_text}.")
        elif box_color == "orange":
             st.warning(f"**{diag}**: This indicates a disturbance in the heart's electrical rhythm. Immediate review recommended.")
        else:
             st.error(f"**{diag}**: This suggests a structural issue or injury pattern (e.g., Infarction/Hypertrophy). CRITICAL attention required.")


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

    # DISPLAY SIGNAL
    if signal_data is not None:
        st.subheader("Patient Vitals (ECG Lead I)")
        
        # Determine shape [12, 5000] vs [5000, 12]
        if signal_data.shape[0] == 12:
            data_t = signal_data.T
        else:
            data_t = signal_data
            
        df_chart = pd.DataFrame(data_t, columns=[f"L{i+1}" for i in range(12)])
        st.line_chart(df_chart["L1"], height=250, color="#E74C3C")
        
        with st.expander("View Full 12-Lead Panel"):
            st.line_chart(df_chart, height=450)
            
        st.divider()
        
        # PREDICT
        with st.spinner("Processing Signal through Neural Hierarchy..."):
            # delay for effect?
            # time.sleep(0.5) 
            result = predictor.predict(signal_data)
            
        display_medical_report(result, mode="Signal")
        
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
            st.image(image, caption="Scanned Document", use_column_width=True)
            
        with c2:
            st.write("### AI Vision Engine")
            
            # Lazy Load Visual Predictor
            @st.cache_resource
            def get_visual_predictor():
                from utils.visual_backend import VisualPredictor
                return VisualPredictor()
            
            try:
                vis_predictor = get_visual_predictor()
                if st.button("Analyze Scan"):
                    with st.spinner("Extracting features & Classifying..."):
                        vis_result = vis_predictor.predict(image)
                        
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
        CardioScan Pro v2.0 | Confidential Medical Device Software | © 2025 PBLS Team
    </div>
    """, 
    unsafe_allow_html=True
)
