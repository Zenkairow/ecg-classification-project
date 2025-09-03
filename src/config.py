# src/config.py

# --- Data Configuration for V1 ---
# This version uses the correct ABSOLUTE path for the local Docker container.
DATA_PATH = '/workspace/project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 100 # V1 used the 100Hz data

# --- Model Configuration ---
NUM_CLASSES = 5 # NORM, MI, STTC, CD, HYP
NUM_LEADS = 12
SIGNAL_LENGTH = 1000 # 10 seconds at 100Hz

# --- Training Configuration (not used by the app, but here for completeness) ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10