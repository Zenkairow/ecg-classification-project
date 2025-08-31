# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Import our custom classes and the config file
from data_loader import ECGDataset
from model import Simple1DCNN
import config

def train_model(class_weights):
    """
    Main function to orchestrate the model training and validation process.
    NOW ACCEPTS CLASS WEIGHTS.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The dataset is loaded once here to get the full size for splitting
    full_dataset = ECGDataset(data_path=config.DATA_PATH, metadata_file=config.METADATA_FILE, sampling_rate=config.SAMPLING_RATE)
    
    torch.manual_seed(42) # Ensure the split is the same every time
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    print("Initializing model...")
    model = Simple1DCNN(
        num_classes=config.NUM_CLASSES, 
        num_leads=config.NUM_LEADS, 
        signal_length=config.SIGNAL_LENGTH
    ).to(device)
    
    # *** KEY CHANGE HERE: Pass the class_weights to the loss function ***
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting training with weighted loss...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Validation]"):
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print("Finished Training")
    
    if not os.path.exists('models'):
        os.makedirs('models')
    # Save this as Version 2 of our model
    torch.save(model.state_dict(), 'models/ecg_model_v2.pth')
    print("Model saved to models/ecg_model_v2.pth")

if __name__ == '__main__':
    # --- Calculate Weights (same as in the notebook) ---
    print("Loading dataset to calculate weights...")
    temp_dataset = ECGDataset(data_path=config.DATA_PATH, metadata_file=config.METADATA_FILE, sampling_rate=config.SAMPLING_RATE)
    class_counts = temp_dataset.metadata['diagnostic_superclass'].value_counts()
    total_samples = len(temp_dataset.metadata)
    class_weights_dict = {class_name: total_samples / count for class_name, count in class_counts.items()}
    
    weights = torch.zeros(len(temp_dataset.label_map))
    for class_name, label_id in temp_dataset.label_map.items():
        weights[label_id] = class_weights_dict.get(class_name, 0)
    
    # Move weights to the correct device before passing them to the training function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)
    
    print(f"Using class weights: {weights}")
    
    # --- Run Training ---
    train_model(class_weights=weights)