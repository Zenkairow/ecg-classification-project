import torch
import pandas as pd
import numpy as np
import wfdb
import os
import ast
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the PTB-XL ECG data.
    """
    def __init__(self, data_path, metadata_file, sampling_rate=100):
        """
        Initializes the dataset.

        Args:
            data_path (str): The absolute path to the dataset directory.
            metadata_file (str): The name of the main metadata CSV file.
            sampling_rate (int): The target sampling rate (100 Hz for low-res, 500 Hz for high-res).
        """
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        
        # Load and process the metadata
        self.metadata = self._load_metadata(metadata_file)
        
        print(f"Dataset initialized with {len(self.metadata)} samples.")

    def _load_metadata(self, metadata_file):
        """Loads and preprocesses the metadata file."""
        df = pd.read_csv(os.path.join(self.data_path, metadata_file))
        
        # Convert scp_codes from string to dictionary
        df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
        
        # Use the appropriate filename column based on the desired sampling rate
        if self.sampling_rate == 100:
            df['filename'] = df['filename_lr']
        elif self.sampling_rate == 500:
            df['filename'] = df['filename_hr']
        else:
            raise ValueError("Sampling rate must be 100 or 500")
            
        # For simplicity in V1, let's map the diagnostic superclasses to integer labels
        df['diagnostic_superclass'] = df.scp_codes.apply(self._get_diagnostic_superclass)
        
        # Create numerical labels
        self.label_map = {label: i for i, label in enumerate(df['diagnostic_superclass'].unique())}
        self.inverse_label_map = {i: label for label, i in self.label_map.items()}
        df['label'] = df['diagnostic_superclass'].map(self.label_map)
        
        print("Label Mapping:", self.label_map)
        
        return df

    def _get_diagnostic_superclass(self, scp_codes):
        # A simple function to get the first diagnostic superclass from the scp_codes dict
        for code in scp_codes.keys():
            if 'NORM' in code:
                return 'NORM' # Normal ECG
            if 'MI' in code:
                return 'MI' # Myocardial Infarction
            if 'STTC' in code:
                return 'STTC' # ST/T Change
            if 'CD' in code:
                return 'CD' # Conduction Disturbance
            if 'HYP' in code:
                return 'HYP' # Hypertrophy
        return 'OTHER' # Other conditions
        
    def _preprocess_signal(self, signal):
        """
        Applies filtering and normalization to the signal.
        """
        # --- Signal Filtering ---
        nyquist = 0.5 * self.sampling_rate
        low = 0.5 / nyquist
        high = 45.0 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal, axis=0)
        
        # --- Normalization ---
        mean = np.mean(filtered_signal, axis=0)
        std = np.std(filtered_signal, axis=0)
        std[std == 0] = 1
        normalized_signal = (filtered_signal - mean) / std
        
        return normalized_signal

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Fetches and preprocesses a single sample from the dataset.
        """
        record_info = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_path, record_info['filename'])
        signal_data, _ = wfdb.rdsamp(file_path)
        processed_signal = self._preprocess_signal(signal_data)
        processed_signal = torch.tensor(processed_signal.T, dtype=torch.float32)
        label = torch.tensor(record_info['label'], dtype=torch.long)
        
        return processed_signal, label

# --- Block for testing the script directly ---
if __name__ == '__main__':
    
    # Using the exact path you have confirmed is correct
    DATA_PATH = '/workspace/project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    METADATA_FILE = 'ptbxl_database.csv'
    
    print("Testing the ECGDataset class...")
    dataset = ECGDataset(data_path=DATA_PATH, metadata_file=METADATA_FILE, sampling_rate=100)
    
    print("\nFetching sample 0...")
    signal, label_id = dataset[0]
    
    print("Sample 0 Signal Shape:", signal.shape)
    print("Sample 0 Label ID:", label_id.item())
    print("Label Name:", dataset.inverse_label_map[label_id.item()])
    
    print("\nTesting with PyTorch DataLoader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    signal_batch, label_batch = next(iter(dataloader))
    print("Signal Batch Shape:", signal_batch.shape)
    print("Label Batch Shape:", label_batch.shape)
    print("Batch Labels:", label_batch)