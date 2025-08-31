# src/config.py

# --- Data Configuration ---
# This is the single source of truth for our data path, corrected for the server environment.
DATA_PATH = '/workspace/ecg-classification-project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 500 # Using 500Hz for the V2 model

# --- Model Configuration ---
NUM_CLASSES = 5 # NORM, MI, STTC, CD, HYP
NUM_LEADS = 12
SIGNAL_LENGTH = 5000 # 10 seconds at 500Hz

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15 # Training for 15 epochs