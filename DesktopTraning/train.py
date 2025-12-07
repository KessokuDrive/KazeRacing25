import torch
import torchvision
from torchvision.models import ResNet18_Weights
import os
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
from DataLoader import HDF5Dataset
import torchvision.transforms as transforms

# Configuration: Set the dataset name
# Users can change this to point to any HDF5 dataset folder (baseline_hdf5, custom_hdf5, etc.)
DATASET = 'baseline_hdf5'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'datasets', DATASET)
HDF5_FILE = os.path.join(DATASET_PATH, 'dataset.h5')



device = torch.device('cuda')


TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_eval(dataloader, model, is_training, epoch, metrics_history):
    """
    Train or evaluate the model and track metrics.
    
    Args:
        dataloader: DataLoader for training/validation data
        model: The neural network model
        is_training: Boolean indicating if in training mode
        epoch: Current epoch number
        metrics_history: Dictionary to store historical metrics
        
    Returns:
        tuple: (average_loss, average_error)
    """
    try:
        if is_training:
            model = model.train()
            mode_name = "Train"
        else:
            model = model.eval()
            mode_name = "Valid"
        
        total_loss = 0.0
        total_error = 0.0
        batch_count = 0
        
        with torch.set_grad_enabled(is_training):
            for batch, (images, xy) in enumerate(dataloader):
                # send data to device
                images = images.to(device)
                xy = xy.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute MSE loss over x, y coordinates
                loss = torch.mean((outputs - xy)**2)
                
                # Calculate L2 error (Euclidean distance)
                error = torch.mean(torch.sqrt(torch.sum((outputs - xy)**2, dim=1)))
                
                total_loss += loss.item()
                total_error += error.item()
                batch_count += 1
                
                current = batch * batch_size + len(images)
                if batch % 10 == 0:
                    print(f"{mode_name} Batch {batch}: loss={loss.item():>7f}, error={error.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
                
                if is_training:
                    # run backpropagation to accumulate gradients
                    loss.backward()
                    # step optimizer to adjust parameters
                    optimizer.step()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_error = total_error / batch_count if batch_count > 0 else 0
        
        # Store metrics in history
        if is_training:
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['train_error'].append(avg_error)
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.7f}, Train Error: {avg_error:.7f}")
        else:
            metrics_history['valid_loss'].append(avg_loss)
            metrics_history['valid_error'].append(avg_error)
            print(f"Epoch {epoch} - Valid Loss: {avg_loss:.7f}, Valid Error: {avg_error:.7f}")
            
            # Save model after validation
            torch.save(model.state_dict(), f"model_{epoch}.pth")
        
        return avg_loss, avg_error
        
    except Exception as e:
        print(f"Error during {mode_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0


def plot_metrics(metrics_history):
    """
    Plot training metrics in real-time using livelossplot.
    
    Args:
        metrics_history: Dictionary containing loss and error history
    """
    if metrics_history['train_loss']:
        logs = {
            'loss': metrics_history['train_loss'][-1],
            'val_loss': metrics_history['valid_loss'][-1],
            'error': metrics_history['train_error'][-1],
            'val_error': metrics_history['valid_error'][-1]
        }
        plotlosses.update(logs)
        plotlosses.send()


#Load model and optimizer
## ResNet 50
# from torchvision.models import Wide_ResNet50_2_Weights
# model = torchvision.models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
# model.fc = torch.nn.Linear(2048, 2).to(device)
## ResNet 18
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
model.fc = torch.nn.Linear(512, 2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

#Set batch size, larger batch sizes will be train faster and stabilize learning
batch_size = 128
#set number of epochs
num_epochs = 20

#Load datasets and create dataloaders
# Check if HDF5 file exists
if not os.path.exists(HDF5_FILE):
    print(f"Error: HDF5 dataset file not found at {HDF5_FILE}")
    print("Please run labelme2Dataset.py first to convert your dataset to HDF5 format")
    exit(1)

print(f"Loading HDF5 dataset from: {HDF5_FILE}")
train_datasets = HDF5Dataset(HDF5_FILE, split='train', transform=TRANSFORMS, random_hflip=True)
valid_datasets = HDF5Dataset(HDF5_FILE, split='valid', transform=TRANSFORMS, random_hflip=True)
train_dataloader = DataLoader(train_datasets, batch_size, shuffle=True)
test_dataloader = DataLoader(valid_datasets, batch_size, shuffle=True)

print(f"Training samples: {len(train_datasets)}")
print(f"Validation samples: {len(valid_datasets)}")


# Initialize metrics history for tracking convergence
metrics_history = {
    'train_loss': [],
    'valid_loss': [],
    'train_error': [],
    'valid_error': []
}

print("Starting training with real-time visualization...")
print(f"Dataset: {DATASET_PATH}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}\n")

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-------------------------------")
    
    # Train epoch
    train_loss, train_error = train_eval(train_dataloader, model, True, epoch, metrics_history)
    
    # Validate epoch
    valid_loss, valid_error = train_eval(test_dataloader, model, False, epoch, metrics_history)
    
    print(f"Convergence: Train Loss={train_loss:.7f}, Valid Loss={valid_loss:.7f}")

# Save final metrics to file
print("\nTraining completed!")
print("Saving final metrics...")
metrics_file = os.path.join(SCRIPT_DIR, 'training_metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("Epoch\tTrain_Loss\tValid_Loss\tTrain_Error\tValid_Error\n")
    for i in range(len(metrics_history['train_loss'])):
        f.write(f"{i+1}\t{metrics_history['train_loss'][i]:.7f}\t{metrics_history['valid_loss'][i]:.7f}\t")
        f.write(f"{metrics_history['train_error'][i]:.7f}\t{metrics_history['valid_error'][i]:.7f}\n")

print(f"Metrics saved to: {metrics_file}")
print("Plot window will remain open. Close it to exit.")
