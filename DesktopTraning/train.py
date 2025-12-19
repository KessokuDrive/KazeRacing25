import torch
import torchvision
import os
import numpy as np
import time
from collections import defaultdict

from torch.utils.data import DataLoader
from DataLoader import XYDataset
import torchvision.transforms as transforms

# Configuration: Set the dataset parent folder path
# Users can easily change this to point to any dataset folder (baseline_processed, custom_processed, etc.)
DATASET = 'baseline_processed'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'datasets', DATASET)

device = torch.device('cuda')


TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_eval(dataloader, model, optimizer, batch_size, is_training, epoch, metrics_history):
    """
    Train or evaluate the model and track metrics.
    
    Args:
        dataloader: DataLoader for training/validation data
        model: The neural network model
        optimizer: Optimizer for training
        batch_size: Batch size for progress display
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
        
        # Debug: Check dataloader
        if len(dataloader) == 0:
            print(f"ERROR: {mode_name} dataloader is empty! Dataset size: {len(dataloader.dataset)}")
            return 0, 0
        
        with torch.set_grad_enabled(is_training):
            for batch, (images, xy) in enumerate(dataloader):
                # send data to device (non_blocking=True speeds up transfer when using pin_memory)
                images = images.to(device, non_blocking=True)
                xy = xy.to(device, non_blocking=True)

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
        
        # Check if any batches were processed
        if batch_count == 0:
            print(f"WARNING: No batches processed during {mode_name}! Check if dataset is empty or dataloader is misconfigured.")
            print(f"Dataset size: {len(dataloader.dataset)}")
            return 0, 0
        
        avg_loss = total_loss / batch_count
        avg_error = total_error / batch_count
        
        # Store metrics in history
        if is_training:
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['train_error'].append(avg_error)
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.7f}, Train Error: {avg_error:.7f} (Processed {batch_count} batches)")
        else:
            metrics_history['valid_loss'].append(avg_loss)
            metrics_history['valid_error'].append(avg_error)
            print(f"Epoch {epoch} - Valid Loss: {avg_loss:.7f}, Valid Error: {avg_error:.7f} (Processed {batch_count} batches)")
            
            # Save model every 5 epochs (optimization: reduce I/O overhead)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"model_{epoch}.pth")
        
        return avg_loss, avg_error
        
    except Exception as e:
        print(f"ERROR during {mode_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batch size: {batch_size}")
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



def main():
    """Main training function - required for Windows multiprocessing."""
    
    #Load model and optimizer
    from torchvision.models import ResNet18_Weights
    ## ResNet 50
    # from torchvision.models import ResNet50_Weights
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

    # Speed optimization: Number of parallel workers for data loading
    # Set to number of CPU cores (typically 4-8). 0 = single-threaded (slow)
    # On Windows, num_workers > 0 requires code to be in if __name__ == '__main__'
    num_workers = 4  # Parallel data loading significantly speeds up training

    #Load datasets and create dataloaders
    train_txt_path = os.path.join(DATASET_PATH, 'train.txt')
    valid_txt_path = os.path.join(DATASET_PATH, 'valid.txt')

    # Check if dataset files exist
    if not os.path.exists(train_txt_path):
        print(f"ERROR: Training dataset file not found: {train_txt_path}")
        print("Please ensure the dataset has been prepared correctly.")
        return

    if not os.path.exists(valid_txt_path):
        print(f"ERROR: Validation dataset file not found: {valid_txt_path}")
        print("Please ensure the dataset has been prepared correctly.")
        return

    train_datasets = XYDataset(train_txt_path, TRANSFORMS, random_hflip=True)
    valid_datasets = XYDataset(valid_txt_path, TRANSFORMS, random_hflip=True)

    # Check if datasets are empty
    if len(train_datasets) == 0:
        print(f"ERROR: Training dataset is empty! Found 0 samples in {train_txt_path}")
        return

    if len(valid_datasets) == 0:
        print(f"ERROR: Validation dataset is empty! Found 0 samples in {valid_txt_path}")
        return

    print(f"Dataset loaded: {len(train_datasets)} training samples, {len(valid_datasets)} validation samples")

    # Optimized DataLoader with parallel loading and pinned memory
    # pin_memory=True: Faster GPU transfer (2-3x speedup for data loading)
    # num_workers: Parallel data loading on CPU
    train_dataloader = DataLoader(train_datasets, batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(valid_datasets, batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)

    # Initialize metrics history for tracking convergence
    metrics_history = {
        'train_loss': [],
        'valid_loss': [],
        'train_error': [],
        'valid_error': [],
        'epoch_time': []  # Time in seconds for each epoch
    }

    print("Starting training with real-time visualization...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers} (parallel data loading)")
    print(f"Pin memory: True (faster GPU transfer)")
    print(f"Epochs: {num_epochs}\n")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-------------------------------")
        
        # Start timing the epoch
        epoch_start_time = time.time()
        
        # Train epoch
        train_loss, train_error = train_eval(train_dataloader, model, optimizer, batch_size, True, epoch, metrics_history)
        
        # Validate epoch
        valid_loss, valid_error = train_eval(test_dataloader, model, optimizer, batch_size, False, epoch, metrics_history)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        metrics_history['epoch_time'].append(epoch_time)
        
        # Format time for display (handle sub-second times properly)
        if epoch_time < 1.0:
            time_str = f"{epoch_time:.2f}s"
        elif epoch_time < 60.0:
            time_str = f"{epoch_time:.2f}s"
        else:
            minutes = int(epoch_time // 60)
            seconds = int(epoch_time % 60)
            time_str = f"{minutes}m {seconds}s"
        
        print(f"Convergence: Train Loss={train_loss:.7f}, Valid Loss={valid_loss:.7f}")
        print(f"Epoch Time: {time_str}")

    # Save final metrics to file
    print("\nTraining completed!")
    print("Saving final metrics...")
    metrics_file = os.path.join(SCRIPT_DIR, 'training_metrics.txt')

    # Calculate total training time
    if metrics_history['epoch_time']:
        total_time = sum(metrics_history['epoch_time'])
        avg_epoch_time = np.mean(metrics_history['epoch_time'])
        min_epoch_time = min(metrics_history['epoch_time'])
        max_epoch_time = max(metrics_history['epoch_time'])
        
        # Format total time
        if total_time < 60.0:
            total_time_str = f"{total_time:.2f} seconds"
        else:
            total_minutes = int(total_time // 60)
            total_seconds = int(total_time % 60)
            total_time_str = f"{total_minutes}m {total_seconds}s ({total_time:.2f} seconds)"
    else:
        total_time = 0.0
        avg_epoch_time = 0.0
        min_epoch_time = 0.0
        max_epoch_time = 0.0
        total_time_str = "0.00 seconds"

    with open(metrics_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tValid_Loss\tTrain_Error\tValid_Error\tEpoch_Time_Seconds\n")
        for i in range(len(metrics_history['train_loss'])):
            epoch_time_val = metrics_history['epoch_time'][i] if i < len(metrics_history['epoch_time']) else 0.0
            f.write(f"{i+1}\t{metrics_history['train_loss'][i]:.7f}\t{metrics_history['valid_loss'][i]:.7f}\t")
            f.write(f"{metrics_history['train_error'][i]:.7f}\t{metrics_history['valid_error'][i]:.7f}\t")
            f.write(f"{epoch_time_val:.2f}\n")

    print(f"Metrics saved to: {metrics_file}")
    print(f"\nTraining Statistics:")
    print(f"  Total Training Time: {total_time_str}")
    print(f"  Average Epoch Time: {avg_epoch_time:.2f} seconds")
    print(f"  Fastest Epoch: {min_epoch_time:.2f} seconds")
    print(f"  Slowest Epoch: {max_epoch_time:.2f} seconds")
    print("Plot window will remain open. Close it to exit.")


if __name__ == '__main__':
    # Required for multiprocessing to work safely on Windows
    # This prevents the RuntimeError when using num_workers > 0
    main()
