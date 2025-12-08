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
# --- Configuration ---
BATCH_SIZE = 4
IMAGE_SIZE = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_CLASSES = 5 # Placeholder, dynamic detection used
DATA_DIR = 'data_synthesis/output/output/images/'
CSV_PATH = 'data/train_labels.csv'
MODEL_SAVE_PATH = 'models/classifier_resnet50_optimized.pth'

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
            
        # Dynamic Class Mapping (String -> Number) based on ALL unique labels
        # Handle case where column might be 'label' or 'diagnostic_superclass'
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        
        unique_labels = sorted(self.annotations[loss_col].unique().tolist())
        self.class_map = {label: i for i, label in enumerate(unique_labels)}
        
        print(f"Dataset Loaded. Total valid images: {len(self.annotations)}")
        print(f"Detected {len(unique_labels)} Classes: {self.class_map}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Get Filename
        img_name = str(self.annotations.iloc[index]['filename'])
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Open Image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Create dummy image for stability
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        # 3. Get Label
        loss_col = 'label' if 'label' in self.annotations.columns else 'diagnostic_superclass'
        label_str = self.annotations.iloc[index][loss_col]
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
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optimized Transforms (Augmentation)
    # Note: No Flip (medical meaning), No Rotation > 10deg
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data Loader
    train_dataset_full = ECGImageDataset(csv_file=CSV_PATH, root_dir=DATA_DIR, transform=None)
    
    if len(train_dataset_full) == 0:
        print("CRITICAL: Dataset is empty! Check CSV labels.")
        return

    # Split 80/20
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])

    # Apply specific transforms to subsets
    class SubsetWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_data = SubsetWrapper(train_subset, train_transform)
    val_data = SubsetWrapper(val_subset, val_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing ResNet50...")
    num_classes_detected = len(train_dataset_full.class_map)
    print(f"Configuring model for {num_classes_detected} classes...")
    
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    # Add Dropout for regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes_detected)
    )
    model = model.to(device)

    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    # Weight Decay for L2 Regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training Loop
    best_val_acc = 0.0
    print(f"Starting optimized training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc="Training")
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            loop.set_postfix(loss=loss.item())
        
        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Results -> Avg Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                val_loss += loss.item()
                
                _, predictions = scores.max(1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Model! Saving to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model()
