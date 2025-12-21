import pandas as pd
import os
import sys

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Imports
from src.config import STAGE2_DATA_PATH

# Output Paths
OUTPUT_RHYTHM = os.path.join(os.path.dirname(STAGE2_DATA_PATH), "stage3_rhythm_dataset.csv")
OUTPUT_STRUCTURE = os.path.join(os.path.dirname(STAGE2_DATA_PATH), "stage3_structure_dataset.csv")

def prepare_stage3_data():
    print("--- Preparing Data for Stage 3 (Specialists) ---")
    
    if not os.path.exists(STAGE2_DATA_PATH):
        print(f"Error: Stage 2 dataset not found at {STAGE2_DATA_PATH}")
        print("Please run src/prepare_stage2_data.py first.")
        return

    print(f"Loading Stage 2 Data from: {STAGE2_DATA_PATH}")
    df = pd.read_csv(STAGE2_DATA_PATH)
    print(f"Total Stage 2 Samples: {len(df)}")
    
    # ---------------------------------------------------------
    # Split Logic: Based on 'router_label' created in Stage 2
    # 0 = Rhythm
    # 1 = Structure
    # ---------------------------------------------------------
    
    # Filter Rhythm
    df_rhythm = df[df['router_label'] == 0].copy()
    
    # Filter Structure
    df_structure = df[df['router_label'] == 1].copy()
    
    # Save Files
    print(f"\nSaving Rhythm Dataset...")
    df_rhythm.to_csv(OUTPUT_RHYTHM, index=False)
    print(f" -> Saved to {OUTPUT_RHYTHM}")
    print(f" -> Count: {len(df_rhythm)} samples")
    
    print(f"\nSaving Structure Dataset...")
    df_structure.to_csv(OUTPUT_STRUCTURE, index=False)
    print(f" -> Saved to {OUTPUT_STRUCTURE}")
    print(f" -> Count: {len(df_structure)} samples")
    
    print("\n--- Stage 3 Preparation Complete ---")
    print("Next Steps:")
    print("1. Train Rhythm Specialist using stage3_rhythm_dataset.csv")
    print("2. Train Structure Specialist using stage3_structure_dataset.csv")

if __name__ == "__main__":
    prepare_stage3_data()
