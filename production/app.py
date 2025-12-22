import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

# Add current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.backend import CardiacPredictor

# Page Config
st.set_page_config(
    page_title="CardioScan AI",
    page_icon="🫀",
    layout="wide"
)

# Title
st.title("🫀 CardioScan AI: Hierarchical ECG Diagnosis")
st.markdown("### Advanced Cardiac Diagnostics using Multi-Stage AI")
st.markdown("---")

# Initialize Predictor (Cached)
@st.cache_resource
def get_predictor():
    return CardiacPredictor()

try:
    predictor = get_predictor()
    st.sidebar.success("Models Loaded Successfully")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Sidebar Control
st.sidebar.header("Input Data")
input_option = st.sidebar.radio("Choose Input Method", ["Upload .npy File", "Load Random Sample"])

signal_data = None
ground_truth = "Unknown"

if input_option == "Upload .npy File":
    uploaded_file = st.sidebar.file_uploader("Upload ECG Signal (.npy)", type=['npy'])
    if uploaded_file is not None:
        signal_data = np.load(uploaded_file)
        st.sidebar.info(f"Loaded: {uploaded_file.name}")

elif input_option == "Load Random Sample":
    # Try to find the dataset path relative to root, or assume local for demo
    # In production, this might point to a specific test folder
    DATA_ROOT = "../data" # Assuming running from production/ folder
    CSV_PATH = "../data/train_labels.csv"
    
    if st.sidebar.button("🎲 Load Random Sample"):
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            # Filter Abnormals for interest
            mask = ~df['label'].isin(['NORM', 'Sinus_Rhythm'])
            if 'diagnostic_superclass' in df.columns:
                 mask = mask & ~df['diagnostic_superclass'].isin(['NORM', 'Sinus_Rhythm'])
            
            sample = df[mask].sample(1).iloc[0]
            
            fname = sample.get('filename_hr')
            if pd.isna(fname): fname = sample.get('filename')
            
            # Sanitization
            if str(fname).endswith('.png'): fname = str(fname).replace('.png', '.npy')
            if not str(fname).endswith('.npy'): fname = str(fname) + '.npy'
            
            fpath = os.path.join(DATA_ROOT, fname)
            
            if os.path.exists(fpath):
                signal_data = np.load(fpath)
                ground_truth = sample['label']
                st.session_state['signal_data'] = signal_data
                st.session_state['ground_truth'] = ground_truth
                st.success(f"Loaded Sample: {fname}")
            else:
                st.error(f"File not found: {fpath}")
        else:
            st.error("Data directory not found relative to app.")

    # Retain state
    if 'signal_data' in st.session_state:
        signal_data = st.session_state['signal_data']
        ground_truth = st.session_state.get('ground_truth', 'Unknown')

# Main Display
if signal_data is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("ECG Signal Visualization (Lead I)")
        # Plot just Lead I for clarity, or all leads? Lead I is usually index 0
        # Signal shape logic: [12, 5000] or [5000, 12]
        if signal_data.shape[0] == 12:
            df_chart = pd.DataFrame(signal_data.T, columns=[f"Lead {i+1}" for i in range(12)])
        else:
            df_chart = pd.DataFrame(signal_data, columns=[f"Lead {i+1}" for i in range(12)])
            
        st.line_chart(df_chart[['Lead 1']], height=300)
        
        with st.expander("Show All 12 Leads"):
             st.line_chart(df_chart, height=500)

    with col2:
        st.subheader("AI Diagnosis")
        
        with st.spinner("Analyzing Heart Rhythm & Structure..."):
            result = predictor.predict(signal_data)
            
        # Display Results
        triage = result['triage']
        diag = result['diagnosis']
        conf = result['confidence']
        top3 = result['top3']
        
        # Color coding
        color = "red" if triage == "Structure" else "orange"
        
        st.info(f"**Triage Route:** {triage} Issue")
        st.metric(label="Primary Diagnosis", value=diag, delta=f"{conf:.1%} Confidence")
        
        st.markdown("### Top Probabilities")
        st.bar_chart(top3)
        
        if ground_truth != "Unknown":
            st.markdown("---")
            st.caption(f"Ground Truth Label: **{ground_truth}**")
            
else:
    st.info("👈 Upload a file or Load a Random Sample to begin.")
