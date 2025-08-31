# src/config.py

# --- Data Configuration ---
# This is the single source of truth for our data path, now with the corrected name.
DATA_PATH = '/workspace/project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 100 # Using 100Hz for faster training

# --- Model Configuration ---
NUM_CLASSES = 5 # NORM, MI, STTC, CD, HYP
NUM_LEADS = 12
SIGNAL_LENGTH = 1000

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10