# System Configuration for ECG Hierarchical Pipeline

# --- Stage 1: The Gatekeeper ---
# Threshold calibrated to ensure > 99% Recall on Validation Set
STAGE1_THRESHOLD = 0.14
STAGE1_MODEL_PATH = "models/hierarchy_stage1_gatekeeper.pth"

# --- Stage 2: The Router ---
STAGE2_MODEL_PATH = "models/hierarchy_stage2_router.pth"
STAGE2_DATA_PATH = "data/stage2_router_dataset.csv"

# --- Stage 3: The Specialists ---
# Rhythm Specialist
STAGE3_RHYTHM_DATA_PATH = "data/stage3_rhythm_dataset.csv"
STAGE3_RHYTHM_MODEL_PATH = "models/hierarchy_stage3_rhythm.pth"
# Structure Specialist
STAGE3_STRUCTURE_DATA_PATH = "data/stage3_structure_dataset.csv"
STAGE3_STRUCTURE_MODEL_PATH = "models/hierarchy_stage3_structure.pth"

# --- V3: AdvancedCardiacNet ---
V3_STRUCTURE_MODEL_PATH = "models/v3_specialist_structure.pth"

# --- Visual Hierarchy (Engine A) ---
VISUAL_ROUTER_DATA_PATH = "data/visual_router_dataset.csv"
VISUAL_RHYTHM_DATA_PATH = "data/visual_rhythm_dataset.csv"
VISUAL_STRUCTURE_DATA_PATH = "data/visual_structure_dataset.csv"
VISUAL_ROUTER_MODEL_PATH = "models/v3_visual_router_1024.pth"
VISUAL_RHYTHM_MODEL_PATH = "models/v3_visual_rhythm_1024.pth"
# Structure was already trained and saved as experimental, let's keep the naming convention clean for documentation
VISUAL_STRUCTURE_MODEL_PATH = "models/v3_visual_structure_1024.pth"

# --- Common ---
SIGNAL_LENGTH = 5000
SAMPLING_RATE = 500
