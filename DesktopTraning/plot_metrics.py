"""
Plot training metrics from training_metrics.txt into two separate plots:
1. Loss plot (Train Loss vs Valid Loss)
2. Error plot (Train Error vs Valid Error)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(SCRIPT_DIR, 'training_metrics.txt')

def plot_metrics():
    # Read the metrics file
    epochs = []
    train_loss = []
    valid_loss = []
    train_error = []
    valid_error = []
    
    with open(METRICS_FILE, 'r') as f:
        lines = f.readlines()
        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 5:
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                valid_loss.append(float(parts[2]))
                train_error.append(float(parts[3]))
                valid_error.append(float(parts[4]))
    
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    train_error = np.array(train_error)
    valid_error = np.array(valid_error)
    
    # Plot 1: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, valid_loss, 'r-', label='Valid Loss', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_output_path = os.path.join(SCRIPT_DIR, 'training_loss.png')
    plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {loss_output_path}")
    plt.close()
    
    # Plot 2: Error
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_error, 'b-', label='Train Error', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, valid_error, 'r-', label='Valid Error', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error (L2 Distance)', fontsize=12)
    plt.title('Training and Validation Error', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    error_output_path = os.path.join(SCRIPT_DIR, 'training_error.png')
    plt.savefig(error_output_path, dpi=300, bbox_inches='tight')
    print(f"Error plot saved to: {error_output_path}")
    plt.close()
    
    print("\nPlots generated successfully!")

if __name__ == '__main__':
    plot_metrics()
