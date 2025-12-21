# System Configuration for ECG Hierarchical Pipeline

# --- Stage 1: The Gatekeeper ---
# Threshold calibrated to ensure > 99% Recall on Validation Set
STAGE1_THRESHOLD = 0.14
STAGE1_MODEL_PATH = "models/hierarchy_stage1_gatekeeper.pth"

# --- Stage 2: The Router ---
STAGE2_MODEL_PATH = "models/hierarchy_stage2_router.pth"
STAGE2_DATA_PATH = "data/stage2_router_dataset.csv"

# --- Common ---
SIGNAL_LENGTH = 5000
SAMPLING_RATE = 500
