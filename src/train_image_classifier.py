import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import sys

# --- Configuration ---
BATCH_SIZE = 8
IMAGE_SIZE = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_CLASSES = 5
DATA_DIR = 'data_synthesis/output/output/images/'
CSV_PATH = 'data/train_labels.csv'
MODEL_SAVE_PATH = 'models/classifier_resnet50.pth'

# --- Dataset Class ---
class ECGImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV
        try:
            self.annotations = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
            
        # Define Class Mapping (String -> Number)
        self.class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
        
        # Filter: Keep only rows where label is in our map
        # (This removes 'Unknown', 'OTHER', or 'NDT' if they exist)
        original_count = len(self.annotations)
        self.annotations = self.annotations[self.annotations['label'].isin(self.class_map.keys())]
        filtered_count = len(self.annotations)
        
        print(f"Dataset Loaded. Kept {filtered_count}/{original_count} valid images.")
        print(f"Classes: {self.class_map}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get Filename (Using 'filename' column, NOT 'filename_hr')
        img_name = str(self.annotations.iloc[index]['filename'])
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Open Image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback: Create a dummy black image to prevent training crash
            # (Better to skip, but this keeps batch size consistent)
            # print(f"Warning: Missing {img_path}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        # 3. Get Label and convert to ID
        label_str = self.annotations.iloc[index]['label']
        label = self.class_map[label_str]

        # 4. Transform
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Training Function ---
def train_model():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data Loader
    dataset = ECGImageDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, transform=transform)
    
    if len(dataset) == 0:
        print("CRITICAL: Dataset is empty after filtering! Check CSV labels.")
        return

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {train_size} images, Validating on {val_size}")

    # Model
    print("Initializing ResNet50...")
    # Use standard weights
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use TQDM for progress bar
        loop = tqdm(train_loader, desc="Training")
        
        for data, targets in loop:
            data = data.to(device)
            targets = targets.to(device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            loop.set_postfix(loss=loss.item())
        
        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1} Results -> Avg Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # Validation Step (Optional but recommended)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                _, predictions = scores.max(1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Validation Acc: {val_acc:.4f}")

        # Save Checkpoint
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)
    train_model()
