# src/app.py

import streamlit as st
import torch
import numpy as np
import wfdb
import os
import ast
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt

# We need to add the src directory to the path to import our custom modules
import sys
sys.path.append('src')
from model_v3 import ResNet1D
import config

# --- App Configuration ---
st.set_page_config(
    page_title="ECG Arrhythmia Classification",
    page_icon="❤️",
    layout="wide"
)

# --- Model and Data Loading ---

@st.cache_resource
def load_model_and_metadata():
    """Loads the trained ResNet model and the label mapping."""
    # This function is now more robust for finding the label map
    temp_df = pd.read_csv(os.path.join(config.DATA_PATH, config.METADATA_FILE))
    temp_df['scp_codes'] = temp_df.scp_codes.apply(lambda x: ast.literal_eval(x))

    def get_diagnostic_superclass(scp_codes):
        for code in scp_codes.keys():
            if 'NORM' in code: return 'NORM'
            if 'MI' in code: return 'MI'
            if 'STTC' in code: return 'STTC'
            if 'CD' in code: return 'CD'
            if 'HYP' in code: return 'HYP'
        return 'OTHER'
    
    temp_df['diagnostic_superclass'] = temp_df.scp_codes.apply(get_diagnostic_superclass)
    label_map = {label: i for i, label in enumerate(temp_df['diagnostic_superclass'].unique())}
    inverse_label_map = {i: label for label, i in label_map.items()}

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1D(num_classes=config.NUM_CLASSES, num_leads=config.NUM_LEADS).to(device)
    
    model_path = 'models/ecg_model_v4.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error(f"Model file not found at {model_path}. Please make sure the model is trained and saved.")
        return None, None
    
    model.eval()
    return model, inverse_label_map

def preprocess_signal(signal, sampling_rate):
    """Applies the same preprocessing steps as used in training."""
    nyquist = 0.5 * sampling_rate
    low = 0.5 / nyquist
    high = 45.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    
    mean = np.mean(filtered_signal, axis=0)
    std = np.std(filtered_signal, axis=0)
    std[std == 0] = 1
    normalized_signal = (filtered_signal - mean) / std
    
    if normalized_signal.shape[0] < config.SIGNAL_LENGTH:
        padding = np.zeros((config.SIGNAL_LENGTH - normalized_signal.shape[0], config.NUM_LEADS))
        normalized_signal = np.vstack((normalized_signal, padding))
    elif normalized_signal.shape[0] > config.SIGNAL_LENGTH:
        normalized_signal = normalized_signal[:config.SIGNAL_LENGTH, :]
        
    return normalized_signal

# --- Main Application ---
st.title("❤️ ECG Arrhythmia Classification")
st.write("Upload a 12-lead ECG file pair (.dat and .hea) to get an AI-powered diagnosis.")

model, inverse_label_map = load_model_and_metadata()

if model is not None:
    # *** KEY CHANGE: Accept multiple files ***
    uploaded_files = st.file_uploader(
        "Choose the ECG file pair (.dat and .hea)", 
        type=["dat", "hea"], 
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) == 2:
        st.write("Files Uploaded! Analyzing...")
        
        # Identify which file is .dat and which is .hea
        dat_file = None
        hea_file = None
        for f in uploaded_files:
            if f.name.endswith('.dat'):
                dat_file = f
            elif f.name.endswith('.hea'):
                hea_file = f
        
        if dat_file and hea_file:
            file_name = dat_file.name.split('.')[0]
            
            # Save both files temporarily
            with open(os.path.join(file_name + ".dat"), "wb") as f_dat:
                f_dat.write(dat_file.getbuffer())
            with open(os.path.join(file_name + ".hea"), "wb") as f_hea:
                f_hea.write(hea_file.getbuffer())

            try:
                # Now that both files exist, wfdb can read the signal
                signal_data, metadata = wfdb.rdsamp(file_name)
                st.success("Successfully read signal data.")
                
                # --- The rest of the prediction logic is the same ---
                fig, ax = plt.subplots(figsize=(15, 4))
                ax.plot(signal_data[:, 0])
                ax.set_title("Uploaded ECG Signal (Lead I)")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude (mV)")
                st.pyplot(fig)

                processed_signal = preprocess_signal(signal_data, metadata['fs'])
                signal_tensor = torch.tensor(processed_signal.T, dtype=torch.float32).unsqueeze(0)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                signal_tensor = signal_tensor.to(device)
                
                with torch.no_grad():
                    output = model(signal_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    predicted_label = inverse_label_map[predicted_idx.item()]

                st.subheader("🔬 AI Diagnosis:")
                st.metric(label="Predicted Condition", value=predicted_label, delta=f"{confidence.item()*100:.2f}% Confidence")

                st.subheader("Confidence Scores")
                prob_df = pd.DataFrame(probabilities.cpu().numpy().T, index=inverse_label_map.values(), columns=['Probability'])
                prob_df.index.name = "Condition"
                st.dataframe(prob_df)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(file_name + ".dat"):
                    os.remove(file_name + ".dat")
                if os.path.exists(file_name + ".hea"):
                    os.remove(file_name + ".hea")
        else:
            st.warning("Please ensure you upload one .dat file and one .hea file.")
    elif uploaded_files:
        st.warning("Please upload exactly two files: one .dat and one .hea.")