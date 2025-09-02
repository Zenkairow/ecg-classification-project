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
from pathlib import Path

# We need to add the src directory to the path to import our custom modules
import sys
sys.path.append('src')
from model_v3 import ResNet1D
import config

# --- App Configuration ---
st.set_page_config(
    page_title="ECG Arrhythmia Classification V2",
    page_icon="❤️",
    layout="wide"
)

# --- Model and Data Loading ---

@st.cache_resource
def load_model_and_metadata():
    """
    Loads the trained ResNet model (V5) and the label mapping.
    """
    # *** KEY CHANGE: Correctly construct the full path to the CSV file ***
    full_csv_path = os.path.join(config.DATA_PATH, config.METADATA_FILE)

    if not os.path.exists(full_csv_path):
        st.error(f"Metadata file not found at {full_csv_path}. Please ensure the DATA_PATH in config.py is correct.")
        return None, None
        
    temp_df = pd.read_csv(full_csv_path)
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
    
    # Dynamically find the project's root directory to build the model path
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    model_path = PROJECT_ROOT / 'models' / 'ecg_model_v5_500hz.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error(f"Model file not found at {model_path}. Please ensure the V5 model is trained and present in the 'models' folder.")
        return None, None
    
    model.eval()
    return model, inverse_label_map

# ... (The rest of the app.py file is the same)
def preprocess_signal(signal, sampling_rate):
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

st.title("❤️ ECG Arrhythmia Classification (V2 Prototype)")
st.write("This prototype uses a ResNet model trained on 500Hz data. Upload a 12-lead ECG file pair (.dat and .hea) to get a diagnosis.")

model, inverse_label_map = load_model_and_metadata()

if model is not None:
    uploaded_files = st.file_uploader("Choose the ECG file pair (.dat and .hea)", type=["dat", "hea"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 2:
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dat_file = hea_file = None
        for f in uploaded_files:
            if f.name.endswith('.dat'): dat_file = f
            elif f.name.endswith('.hea'): hea_file = f
        
        if dat_file and hea_file:
            file_name = dat_file.name.split('.')[0]
            dat_path = os.path.join(temp_dir, dat_file.name)
            hea_path = os.path.join(temp_dir, hea_file.name)
            with open(dat_path, "wb") as f_dat: f_dat.write(dat_file.getbuffer())
            with open(hea_path, "wb") as f_hea: f_hea.write(hea_file.getbuffer())

            try:
                signal_data, metadata = wfdb.rdsamp(os.path.join(temp_dir, file_name))
                st.success("Successfully read signal data.")
                
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
                if os.path.exists(dat_path): os.remove(dat_path)
                if os.path.exists(hea_path): os.remove(hea_path)
        else:
            st.warning("Please ensure you upload one .dat file and one .hea file.")
    elif uploaded_files:
        st.warning("Please upload exactly two files: one .dat and one .hea.")