# src/config.py
import os
from pathlib import Path

# --- Data Configuration ---
# This version dynamically finds the project root and constructs absolute paths.
# This makes it robust and independent of the script's running location.

# Dynamically find the project's root directory.
# Path(__file__) is the path to this config.py file.
# .parent gives us the 'src' folder.
# .parent.parent gives us the main project root (e.g., 'ecg_classification').
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Construct a full, absolute path to the data directory.
DATA_PATH ='data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv'

METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 500

# --- Model Configuration ---
NUM_CLASSES = 5
NUM_LEADS = 12
SIGNAL_LENGTH = 5000

# --- Training Configuration ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15