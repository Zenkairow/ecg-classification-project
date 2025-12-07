import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import ast
import os
import sys

# --- Configuration ---
BATCH_SIZE = 8
IMAGE_SIZE = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_CLASSES = 5
DATA_DIR = 'data_synthesis/output/signal_targets_final/'
LABELS_FILE = 'data/ptbxl_database.csv'
MODEL_SAVE_PATH = 'models/classifier_resnet50.pth'

# --- Dataset Class ---
class ECGImageDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load metadata
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found at {labels_file}. Assuming remote execution.")
            # Create a dummy dataframe for syntax checking if file doesn't exist locally
            self.metadata = pd.DataFrame(columns=['filename_hr', 'scp_codes'])
        else:
            self.metadata = pd.read_csv(labels_file)

        # Preprocessing similar to data_loader.py
        if 'scp_codes' in self.metadata.columns:
            self.metadata['scp_codes'] = self.metadata['scp_codes'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # Map Diagnostic Superclass
            self.metadata['diagnostic_superclass'] = self.metadata.scp_codes.apply(self._get_diagnostic_superclass)
            
            # Label Mapping
            self.label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4, 'OTHER': 5}
            # Filter out 'OTHER' if we only want the main 5, or include it. 
            # Context implies 5 classes usually, but let's stick to the map.
            # If we strictly want 5 classes, we might drop OTHER or map it to one.
            # For now, let's map 'OTHER' to -1 or filter.
            # Let's assume we filter for now to be safe with 5 output neurons.
            self.metadata = self.metadata[self.metadata['diagnostic_superclass'] != 'OTHER']
            
            self.metadata['label'] = self.metadata['diagnostic_superclass'].map(self.label_map)
        
        # Filename adjustment (Assuming filename_hr matches the image name)
        # We need to verify if images have extensions in the CSV or not.
        # Usually PTB-XL CSV filenames do not have extensions.
        # Images in data_synthesis/output/signal_targets_final/ are likely .png
        
    def _get_diagnostic_superclass(self, scp_codes):
        for code in scp_codes.keys():
            if 'NORM' in code: return 'NORM'
            if 'MI' in code: return 'MI'
            if 'STTC' in code: return 'STTC'
            if 'CD' in code: return 'CD'
            if 'HYP' in code: return 'HYP'
        return 'OTHER'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Construct path - Assuming local filename in CSV matches image basename
        # We try adding .png if not present
        img_name = row['filename_hr']
        if not str(img_name).endswith('.png'):
            img_name = str(img_name) + '.png'
            
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback for training stability - duplicate random item or error?
            # For this script, we'll error to be loud about missing data
            raise FileNotFoundError(f"Image not found: {img_path}")

        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label

# --- Training Function ---
def train_model():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    print(f"Loading data from {DATA_DIR} with labels {LABELS_FILE}...")
    dataset = ECGImageDataset(DATA_DIR, LABELS_FILE, transform=transform)
    
    # Check if dataset is empty (local dev mode)
    if len(dataset) == 0:
        print("Dataset is empty (Check paths). Exiting safely.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model - ResNet50
    print("Initializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify Final Layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}] Complete. Avg Loss: {epoch_loss:.4f}")
        
    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
