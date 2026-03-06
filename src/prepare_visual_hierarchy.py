"""
Prepare Visual Hierarchy Data — Split visual dataset by Rhythm/Structure taxonomy
================================================================================
Reads the existing visual dataset (data/train_labels.csv, data_synthesis images)
and splits it into visual_router_dataset.csv, visual_rhythm_dataset.csv,
and visual_structure_dataset.csv using the v2.0 taxonomy.
"""

import pandas as pd
import os
import sys

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.engine_a_visual.train_visual_model import CSV_PATH, DATA_DIR
from src.engine_a_visual.train_visual_model import ECGImageDataset

# Taxonomy — same groups as Engine B (prepare_stage2_data.py)
CLASS_0_RHYTHM = [
    'Atrial_Fibrillation', 'SVT', 'AV_Block', 'PVC', 'Paced', 
    'Fascicular_Block', 'NDT', 'WPW', 'DIG', 'EL', 'Dig',
    'Sinus_Rhythm',
]

CLASS_1_STRUCTURE = [
    'MI_Anterior', 'MI_Inferior', 'MI_Lateral', 'LBBB', 'RBBB', 
    'Left_Hypertrophy', 'Right_Hypertrophy', 'Ischemia', 
    'NonSpecific_ST', 'IVCD', 'LNGQT',
]

IGNORE_CLASSES = ['NORM']


def group_visual_label(label):
    """Apply the same clinical taxonomy used across both engines."""
    label = str(label)
    if label in ['AMI', 'ALMI', 'ASMI', 'INJAL', 'INJAS']: return 'MI_Anterior'
    if label in ['IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIL', 'INJIN']: return 'MI_Inferior'
    if label in ['LMI', 'INJLA', 'PMI']: return 'MI_Lateral'
    if 'ISC' in label or label == 'NST_': return 'Ischemia'
    if label in ['CLBBB', 'ILBBB']: return 'LBBB'
    if label in ['CRBBB', 'IRBBB']: return 'RBBB'
    if label == 'IVCD': return 'IVCD'
    if label in ['1AVB', '2AVB', '3AVB']: return 'AV_Block'
    if label in ['LVH', 'LAO/LAE']: return 'Left_Hypertrophy'
    if label in ['RVH', 'RAO/RAE', 'SEHYP']: return 'Right_Hypertrophy'
    if label in ['LAFB', 'LPFB']: return 'Fascicular_Block'
    if label in ['AFIB', 'AFLT']: return 'Atrial_Fibrillation'
    if label in ['SARRH', 'STACH', 'SBRAD', 'SR']: return 'Sinus_Rhythm'
    if label == 'PACE': return 'Paced'
    if label in ['PSVT', 'SVT']: return 'SVT'
    if label == 'NORM': return 'NORM'
    return label


def prepare_visual_hierarchy():
    print("=" * 60)
    print("Preparing Visual Hierarchy Dataset")
    print("=" * 60)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Source CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Original Visual Dataset Size: {len(df)}")
    
    label_col = 'label' if 'label' in df.columns else 'diagnostic_superclass'
    
    router_rows = []
    rhythm_rows = []
    structure_rows = []
    
    stats = {'Rhythm': 0, 'Structure': 0, 'Ignored': 0, 'Unknown': 0}
    
    for idx, row in df.iterrows():
        raw_label = str(row[label_col])
        grouped = group_visual_label(raw_label)
        
        if grouped in IGNORE_CLASSES:
            stats['Ignored'] += 1
            continue
        
        # Check filename validity
        fname = row.get('filename')
        if pd.isna(fname) or str(fname).lower() == 'nan' or str(fname) == '':
            continue
        fname = str(fname)
        
        # Check image exists
        img_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(img_path):
            continue
        
        # Route
        if grouped in CLASS_0_RHYTHM:
            router_label = 0
            stats['Rhythm'] += 1
            rhythm_rows.append({
                'filename': fname,
                'label': raw_label,
                'grouped_label': grouped,
                'router_label': 0,
            })
        elif grouped in CLASS_1_STRUCTURE:
            router_label = 1
            stats['Structure'] += 1
            structure_rows.append({
                'filename': fname,
                'label': raw_label,
                'grouped_label': grouped,
                'router_label': 1,
            })
        else:
            stats['Unknown'] += 1
            continue
        
        router_rows.append({
            'filename': fname,
            'label': raw_label,
            'grouped_label': grouped,
            'router_label': router_label,
        })
    
    # Save
    data_dir = os.path.dirname(CSV_PATH)
    
    router_df = pd.DataFrame(router_rows)
    router_path = os.path.join(data_dir, "visual_router_dataset.csv")
    router_df.to_csv(router_path, index=False)
    
    rhythm_df = pd.DataFrame(rhythm_rows)
    rhythm_path = os.path.join(data_dir, "visual_rhythm_dataset.csv")
    rhythm_df.to_csv(rhythm_path, index=False)
    
    structure_df = pd.DataFrame(structure_rows)
    structure_path = os.path.join(data_dir, "visual_structure_dataset.csv")
    structure_df.to_csv(structure_path, index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Router Dataset:    {len(router_df)} samples -> {router_path}")
    print(f"Rhythm Dataset:    {len(rhythm_df)} samples -> {rhythm_path}")
    print(f"Structure Dataset: {len(structure_df)} samples -> {structure_path}")
    print(f"\nStats:")
    print(f"  Rhythm:    {stats['Rhythm']}")
    print(f"  Structure: {stats['Structure']}")
    print(f"  Ignored:   {stats['Ignored']}")
    print(f"  Unknown:   {stats['Unknown']}")


if __name__ == "__main__":
    prepare_visual_hierarchy()
