# src/config.py
import os
from pathlib import Path

# --- Data Configuration ---
# This is the final, robust version for both local and server environments.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Correct local path (nested)
local_data_path = PROJECT_ROOT / 'data' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

# Correct server path (single-level)
server_data_path = '/workspace/ecg-classification-project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

# Check for a "SERVER_ENV" environment variable to decide which path to use.
if os.getenv('SERVER_ENV') == 'True':
    DATA_PATH = server_data_path
else:
    DATA_PATH = str(local_data_path)

METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 500

# --- Model/Training Configuration ---
NUM_CLASSES = 5
NUM_LEADS = 12
SIGNAL_LENGTH = 5000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15