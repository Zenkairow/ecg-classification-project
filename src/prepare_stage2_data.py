import pandas as pd
import os
import sys

# Setup Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'engine_b_signal'))

from src.engine_b_signal.train_signal_model import CSV_PATH, group_diagnostic_classes

# Specific Mapping from User Directive
CLASS_0_RHYTHM = [
    'Atrial_Fibrillation', 'SVT', 'AV_Block_1st_Deg', 'AV_Block_2nd_Deg', 
    'AV_Block_3rd_Deg', 'PVC', 'Paced', 'Fascicular_Block', 'NDT', 
    'WPW', 'DIG', 'EL', 'Dig' # 'Dig' and 'DIG' handled case-insensitively ideally, but sticking to list
]

CLASS_1_STRUCTURE = [
    'MI_Anterior', 'MI_Inferior', 'MI_Lateral', 'LBBB', 'RBBB', 
    'Left_Hypertrophy', 'Right_Hypertrophy', 'Ischemia', 
    'NonSpecific_ST', 'IVCD', 'LNGQT'
]

# Normal classes to exclude
IGNORE_CLASSES = ['NORM', 'Sinus_Rhythm']

def prepare_stage2_data():
    print("--- Preparing Data for Stage 2 (Router) ---")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Source CSV not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"Original Dataset Size: {len(df)}")
    
    # Identify Label Column (Handle official PTB-XL scp_codes)
    if 'label' in df.columns:
        label_col = 'label'
    elif 'diagnostic_superclass' in df.columns:
        label_col = 'diagnostic_superclass'
    elif 'scp_codes' in df.columns:
        label_col = 'scp_codes'
    else:
        print("Error: Could not find label/scp_codes column in CSV")
        return
    
    filtered_rows = []
    
    stats = {'Rhythm': 0, 'Structure': 0, 'Ignored': 0, 'Unknown': 0}
    
    import ast
    
    for idx, row in df.iterrows():
        raw_val = str(row[label_col])
        
        # Handle PTB-XL scp_codes dictionary format: "{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}"
        if label_col == 'scp_codes':
            try:
                # Convert string representation of dict to actual dict
                codes = ast.literal_eval(raw_val)
                # Sort by likelihood (value) and take the highest one, fallback to first key
                if codes:
                    raw_label = max(codes.keys(), key=lambda k: codes[k])
                else:
                    raw_label = "UNKNOWN"
            except:
                raw_label = "UNKNOWN"
        else:
            raw_label = raw_val
            
        group = group_diagnostic_classes(raw_label)
        
        if group in IGNORE_CLASSES:
            stats['Ignored'] += 1
            continue
            
        # 1.5. Check Filename Validity
        # Try multiple possible column names
        f_name = row.get('filename')
        f_hr = row.get('filename_hr')
        f_lr = row.get('filename_lr')
        
        # Determine the best available filename
        final_filename = None
        
        if isinstance(f_hr, str) and len(f_hr) > 0 and str(f_hr).lower() != 'nan':
            final_filename = f_hr
        elif isinstance(f_lr, str) and len(f_lr) > 0 and str(f_lr).lower() != 'nan':
            final_filename = f_lr
        elif isinstance(f_name, str) and len(f_name) > 0 and str(f_name).lower() != 'nan':
             final_filename = f_name
        # Fallback for non-string types that might be valid (e.g. integers as filenames)
        elif f_name is not None and str(f_name).lower() != 'nan':
             final_filename = str(f_name)
             
        if final_filename is None:
            stats['MissingFile'] = stats.get('MissingFile', 0) + 1
            continue
            
        # Robust extension handling (Match train_signal_model.py logic)
        if final_filename.endswith('.png'):
            final_filename = final_filename.replace('.png', '.npy')
        
        # 2. Map to 0/1
            
        # 2. Map to 0/1
        router_label = -1
        
        # Case insensitive check might be safer given 'Dig' vs 'DIG'
        if group in CLASS_0_RHYTHM:
            router_label = 0
            stats['Rhythm'] += 1
        elif group in CLASS_1_STRUCTURE:
            router_label = 1
            stats['Structure'] += 1
        else:
            # Fallback/Unknown
            # print(f"Warning: Unknown/Unmapped Class: {group} (Raw: {raw_label})")
            stats['Unknown'] += 1
            continue
            
        # Add to new list
        # Keep necessary columns: filename/path and the new label
        # Explicitly copy relevant columns to avoid ambiguity
        new_row = {
            'router_label': router_label,
            'filename_hr': final_filename, # Normalized filename
            'filename_lr': '', # Deprecated/Unused in this new schema
            'age': row.get('age', 0),
            'sex': row.get('sex', 0),
            'label': row.get('label', ''),
            'diagnostic_superclass': row.get('diagnostic_superclass', '')
        }
        filtered_rows.append(new_row)
        
    # Create new DF
    new_df = pd.DataFrame(filtered_rows)
    
    # Save
    output_path = os.path.join(os.path.dirname(CSV_PATH), "stage2_router_dataset.csv")
    new_df.to_csv(output_path, index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Saved to: {output_path}")
    print(f"Total Rows: {len(new_df)}")
    print(f"Stats:")
    print(f"  - Class 0 (Rhythm):    {stats['Rhythm']}")
    print(f"  - Class 1 (Structure): {stats['Structure']}")
    print(f"  - Ignored (Normal):    {stats['Ignored']}")
    print(f"  - Missing Filename:    {stats.get('MissingFile', 0)}")
    print(f"  - Unknown (Dropped):   {stats['Unknown']}")

if __name__ == "__main__":
    prepare_stage2_data()
