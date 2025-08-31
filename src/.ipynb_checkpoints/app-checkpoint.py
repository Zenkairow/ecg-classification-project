# app.py

import streamlit as st
import torch
import numpy as np
import wfdb
import os
from scipy.signal import butter, filtfilt
import pandas as pd

# We need to add the src directory to the path to import our custom modules
import sys
sys.path.append('src')
from model_v3 import ResNet1D # We use the ResNet architecture from model_v3
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
    """
    Loads the trained ResNet model and the label mapping.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    # Load the label map
    # This is a bit of a workaround to get the map without instantiating the full dataset
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
    """
    Applies the same preprocessing steps as used in training.
    """
    # Filter and normalize
    nyquist = 0.5 * sampling_rate
    low = 0.5 / nyquist
    high = 45.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    
    mean = np.mean(filtered_signal, axis=0)
    std = np.std(filtered_signal, axis=0)
    std[std == 0] = 1
    normalized_signal = (filtered_signal - mean) / std
    
    # Pad or truncate to the required signal length
    if normalized_signal.shape[0] < config.SIGNAL_LENGTH:
        padding = np.zeros((config.SIGNAL_LENGTH - normalized_signal.shape[0], config.NUM_LEADS))
        normalized_signal = np.vstack((normalized_signal, padding))
    elif normalized_signal.shape[0] > config.SIGNAL_LENGTH:
        normalized_signal = normalized_signal[:config.SIGNAL_LENGTH, :]
        
    return normalized_signal

# --- Main Application ---

st.title("❤️ ECG Arrhythmia Classification")
st.write("Upload a 12-lead ECG file (.dat) to get an AI-powered diagnosis.")

# Load model and metadata
model, inverse_label_map = load_model_and_metadata()

if model is not None:
    uploaded_file = st.file_uploader("Choose an ECG file (.dat)", type=["dat"])

    if uploaded_file is not None:
        st.write("File Uploaded! Analyzing...")
        
        # We need both .dat and .hea files to read the signal
        # For simplicity, we assume the .hea file has the same name
        file_name = uploaded_file.name.split('.')[0]
        
        # Save uploaded file temporarily to read it with wfdb
        with open(os.path.join(file_name + ".dat"), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # This is a major simplification: we assume a .hea file exists.
        # A real app would need a more robust way to get header info.
        try:
            # Try to read the signal with a dummy header or assume a standard one
            signal_data, metadata = wfdb.rdsamp(file_name)
            st.success("Successfully read signal data.")
            
            # Display a plot of the first lead
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(signal_data[:, 0])
            ax.set_title("Uploaded ECG Signal (Lead I)")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude (mV)")
            st.pyplot(fig)

            # Preprocess the signal
            processed_signal = preprocess_signal(signal_data, metadata['fs'])
            
            # Convert to tensor and add batch dimension
            signal_tensor = torch.tensor(processed_signal.T, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            signal_tensor = signal_tensor.to(device)
            
            with torch.no_grad():
                output = model(signal_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_label = inverse_label_map[predicted_idx.item()]

            st.subheader("🔬 AI Diagnosis:")
            st.metric(label="Predicted Condition", value=predicted_label, delta=f"{confidence.item()*100:.2f}% Confidence")

            # Display confidence scores for all classes
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame(probabilities.cpu().numpy().T, index=inverse_label_map.values(), columns=['Probability'])
            prob_df.index.name = "Condition"
            st.dataframe(prob_df)

            # Clean up temporary files
            os.remove(file_name + ".dat")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("This prototype requires a corresponding .hea (header) file to be present in the same directory to read the signal correctly. This feature is not supported in the live demo.")
            # Clean up in case of error
            if os.path.exists(file_name + ".dat"):
                os.remove(file_name + ".dat")