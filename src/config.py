# src/config.py
import os
from pathlib import Path

# --- Data Configuration ---
# This is the final, robust version. It dynamically finds the project root
# and constructs the correct absolute path to the DATA DIRECTORY.

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# This is the correct local path to the DIRECTORY containing the data
local_data_path = PROJECT_ROOT / 'data' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

# We will use the local path for now. The server logic can be added back if needed.
DATA_PATH = str(local_data_path)

METADATA_FILE = 'ptbxl_database.csv' # This is the FILENAME
SAMPLING_RATE = 500

# --- Model Configuration ---
NUM_CLASSES = 5
NUM_LEADS = 12
SIGNAL_LENGTH = 5000

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15