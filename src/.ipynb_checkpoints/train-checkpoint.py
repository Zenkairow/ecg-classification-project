import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Import our custom classes from other files in the 'src' directory
from data_loader import ECGDataset
from model import Simple1DCNN

# --- Configuration & Hyperparameters ---
# Data paths (ensure this is the correct absolute path inside the container)
DATA_PATH = '/workspace/project/data/ptb-xl-a-large-publicly-available-clinical-electrocardiography-dataset-1.0.3/'
METADATA_FILE = 'ptbxl_database.csv'
SAMPLING_RATE = 100 # Using 100Hz for faster training in the baseline

# Model parameters
NUM_CLASSES = 5 # NORM, MI, STTC, CD, HYP
NUM_LEADS = 12
SIGNAL_LENGTH = 1000 # 10 seconds at 100 Hz

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10 # We'll start with 10 epochs to see how it performs

def train_model():
    """
    Main function to orchestrate the model training and validation process.
    """
    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Create Dataset and DataLoaders ---
    print("Loading and splitting dataset...")
    dataset = ECGDataset(data_path=DATA_PATH, metadata_file=METADATA_FILE, sampling_rate=SAMPLING_RATE)
    
    # Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- 3. Initialize Model, Loss Function, and Optimizer ---
    print("Initializing model...")
    model = Simple1DCNN(num_classes=NUM_CLASSES, num_leads=NUM_LEADS, signal_length=SIGNAL_LENGTH).to(device)
    
    # Loss function - CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - Adam is a popular and effective choice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        # Using tqdm for a progress bar
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            # Move data to the configured device (GPU or CPU)
            signals, labels = signals.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(signals)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)

        # --- 5. Validation Loop ---
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # We don't need to calculate gradients during validation
            for signals, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                signals, labels = signals.to(device), labels.to(device)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print("Finished Training")
    
    # --- 6. Save the trained model ---
    # It's good practice to save the model's state dictionary
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/ecg_model_v1.pth')
    print("Model saved to models/ecg_model_v1.pth")

if __name__ == '__main__':
    train_model()