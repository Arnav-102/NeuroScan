import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy

# Configuration
DATA_DIR = os.path.join("data")  # Expects 'Training' and 'Testing' subfolders
MODEL_SAVE_PATH = "brain_tumor_resnet18.pth"
CHECKPOINT_FILE = "checkpoint.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def train_model():
    # 1. Device Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Augmentation (Expanding dataset effectively to >10k samples)
    # We use aggressive augmentation to improve generalization and robustness
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets
    # Check if data exists
    if not os.path.exists(os.path.join(DATA_DIR, 'Training')):
        print(f"Error: Dataset not found at {DATA_DIR}. Please download the Kaggle dataset.")
        return

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'Training'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'Testing'), val_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Training on {dataset_sizes['train']} images, Validating on {dataset_sizes['val']} images.")
    
    # 4. Initialize ResNet18 Model
    model = models.resnet18(pretrained=True)
    
    # Replace last layer for our 4 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    model = model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 6. Training Loop (with Checkpointing)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_epoch = 0

    # Auto-Resume Logic
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Found checkpoint: {CHECKPOINT_FILE}. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting fresh training.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model and SAVE immediately
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New Best Model Saved! ({best_acc:.4f})")

        # Save Checkpoint after every epoch
        print(f"Saving checkpoint to {CHECKPOINT_FILE}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc # Persistent best_acc across sessions
        }, CHECKPOINT_FILE)
        
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
