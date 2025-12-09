"""
Benchmark script for fast inference on image sequences.

This script reads images from a folder, runs inference as fast as possible,
and calculates FPS statistics. It also saves 1/10 of the processed images
with a green point indicating the inference coordinates.

Usage:
    python3 benchmark.py --model ../model_trt/model_20_trt.pth --images_folder demotrack/ --model_type resnet18 --output_folder ./output
"""

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import PIL.Image
import os
import argparse
import time
import glob

# Try to import TRT module (optional, for Jetson Nano)
try:
    from torch2trt import TRTModule
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def preprocess(image):
    """
    Preprocess image for model inference.
    Converts BGR image to normalized tensor.
    
    Args:
        image: BGR image (numpy array from cv2)
        
    Returns:
        Preprocessed tensor ready for model input
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image_pil = PIL.Image.fromarray(image_rgb)
    # Convert to tensor and normalize
    image_tensor = transforms.functional.to_tensor(image_pil).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
    image_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image_tensor[None, ...]


def detect_model_type_from_checkpoint(model_path):
    """
    Detect model architecture type from checkpoint by examining layer shapes.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Detected model type string or None if cannot determine
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if it's a state_dict or a full checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check layer1.0.conv1.weight shape to determine architecture
        if 'layer1.0.conv1.weight' in state_dict:
            conv1_shape = state_dict['layer1.0.conv1.weight'].shape
            
            # ResNet18/34: [64, 64, 3, 3] in layer1
            # ResNet50: [64, 64, 1, 1] in layer1 (bottleneck)
            # Wide ResNet50_2: [128, 64, 1, 1] in layer1 (wider bottleneck)
            
            if len(conv1_shape) == 4:
                if conv1_shape[0] == 128 and conv1_shape[1] == 64:
                    return 'wide_resnet50_2'
                elif conv1_shape[0] == 64 and conv1_shape[1] == 64:
                    # Check fc layer to distinguish ResNet18 from ResNet50
                    if 'fc.weight' in state_dict:
                        fc_shape = state_dict['fc.weight'].shape
                        if fc_shape[1] == 512:
                            return 'resnet18'
                        elif fc_shape[1] == 2048:
                            return 'resnet50'
                    # Fallback: check if it's a bottleneck (1x1 conv) or regular (3x3 conv)
                    if conv1_shape[2] == 1 and conv1_shape[3] == 1:
                        return 'resnet50'  # Bottleneck architecture
                    else:
                        return 'resnet18'  # Basic block architecture
        
        # Check fc layer as fallback
        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            if fc_shape[1] == 512:
                return 'resnet18'
            elif fc_shape[1] == 2048:
                # Could be ResNet50 or Wide ResNet50_2, check layer1
                if 'layer1.0.conv1.weight' in state_dict:
                    conv1_shape = state_dict['layer1.0.conv1.weight'].shape
                    if conv1_shape[0] == 128:
                        return 'wide_resnet50_2'
                return 'resnet50'
        
        return None
    except Exception as e:
        print(f"Warning: Could not detect model type from checkpoint: {e}")
        return None


def load_model(model_path, model_type='auto', use_trt=True):
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        model_type: Model architecture ('auto', 'resnet18', 'resnet50', 'wide_resnet50_2')
                   If 'auto', will try to detect from checkpoint
        use_trt: Whether to load TRT optimized model
        
    Returns:
        Loaded model in eval mode
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_trt and TRT_AVAILABLE:
        print(f"Loading TRT optimized model from {model_path}")
        model = TRTModule()
        model.load_state_dict(torch.load(model_path))
        return model
    
    # Auto-detect model type if requested
    if model_type == 'auto':
        print("Auto-detecting model architecture from checkpoint...")
        detected_type = detect_model_type_from_checkpoint(model_path)
        if detected_type:
            model_type = detected_type
            print(f"Detected model type: {model_type}")
        else:
            print("Warning: Could not auto-detect model type, trying resnet18...")
            model_type = 'resnet18'
    
    # Load regular PyTorch model
    print(f"Loading {model_type} model from {model_path}")
    
    # Check torchvision version to use correct API
    # Older versions (< 0.13) use pretrained, newer versions use weights
    try:
        # Try new API first (torchvision >= 0.13)
        if model_type == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
            model.fc = torch.nn.Linear(512, 2)
        elif model_type == 'resnet50':
            model = torchvision.models.resnet50(weights=None)
            model.fc = torch.nn.Linear(2048, 2)
        elif model_type == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(weights=None)
            model.fc = torch.nn.Linear(2048, 2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except TypeError:
        # Fall back to old API (torchvision < 0.13) for Jetson Nano compatibility
        if model_type == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(512, 2)
        elif model_type == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(2048, 2)
        elif model_type == 'wide_resnet50_2':
            model = torchvision.models.wide_resnet50_2(pretrained=False)
            model.fc = torch.nn.Linear(2048, 2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights with better error handling
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully!")
    except RuntimeError as e:
        print(f"\nError loading model: {e}")
        print("\nTrying to auto-detect correct model architecture...")
        
        # Try all model types
        model_types = ['resnet18', 'resnet50', 'wide_resnet50_2']
        for try_type in model_types:
            if try_type == model_type:
                continue  # Skip the one we already tried
            try:
                print(f"Trying {try_type}...")
                # Recreate model with different architecture
                try:
                    if try_type == 'resnet18':
                        model = torchvision.models.resnet18(weights=None)
                        model.fc = torch.nn.Linear(512, 2)
                    elif try_type == 'resnet50':
                        model = torchvision.models.resnet50(weights=None)
                        model.fc = torch.nn.Linear(2048, 2)
                    elif try_type == 'wide_resnet50_2':
                        model = torchvision.models.wide_resnet50_2(weights=None)
                        model.fc = torch.nn.Linear(2048, 2)
                except TypeError:
                    # Old API
                    if try_type == 'resnet18':
                        model = torchvision.models.resnet18(pretrained=False)
                        model.fc = torch.nn.Linear(512, 2)
                    elif try_type == 'resnet50':
                        model = torchvision.models.resnet50(pretrained=False)
                        model.fc = torch.nn.Linear(2048, 2)
                    elif try_type == 'wide_resnet50_2':
                        model = torchvision.models.wide_resnet50_2(pretrained=False)
                        model.fc = torch.nn.Linear(2048, 2)
                
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device)
                model.eval()
                print(f"Successfully loaded as {try_type}!")
                return model
            except RuntimeError:
                continue
        
        # If all failed, raise original error
        raise RuntimeError(f"Could not load model. Please specify correct --model_type. Original error: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model


def save_output_image(image, output, frame_number, output_folder, raw_output=None):
    """
    Save image with inference coordinates marked as a green point.
    
    Args:
        image: Original BGR image
        output: Model output (inference coordinates or confidence values)
        frame_number: Current frame number for naming
        output_folder: Path to save output images
        raw_output: Raw output values for display
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create a copy to draw on
    output_image = image.copy()
    
    # Display raw output values in top left corner
    if raw_output is not None:
        raw_text = f"Raw output: {raw_output}"
        cv2.putText(output_image, raw_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Try to convert output to coordinates if it looks like coordinates
    try:
        if len(output) >= 2:
            # Assume first two values are x, y in some coordinate system
            x_raw = output[0]
            y_raw = output[1]
            
            # Display raw values
            coord_text = f"Model output: x={x_raw:.4f}, y={y_raw:.4f}"
            cv2.putText(output_image, coord_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Convert raw outputs (-1 to 0 range) to pixel coordinates
            # Assuming raw outputs are normalized to [-1, 0] range
            # We need to map -1 to 0 and 0 to width-1/height-1
            
            # FIXED: Properly scale from [-1, 0] to [0, width-1]
            # x_raw is between -1 and 0, so x_raw+1 gives us [0, 1]
            # Then multiply by (width-1) to get [0, width-1]
            x_scaled = (x_raw + 1) * (width - 1)
            y_scaled = (y_raw + 1) * (height - 1)
            
            # Convert to integers
            x = int(np.clip(x_scaled, 0, width - 1))
            y = int(np.clip(y_scaled, 0, height - 1))
            
            # Alternative interpretation if the above doesn't work:
            # If your model outputs are actually in [-0.5, 0.5] range, use:
            # x = width // 2 + int(x_raw * width)
            # y = height // 2 + int(y_raw * height)
            
            # Draw green circle at the inferred coordinates
            cv2.circle(output_image, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
            
            # Draw crosshair for better visibility
            cv2.line(output_image, (x-15, y), (x+15, y), (0, 255, 0), 2)
            cv2.line(output_image, (x, y-15), (x, y+15), (0, 255, 0), 2)
            
            # Add text with pixel coordinates
            pixel_text = f"Pixel: ({x}, {y})"
            cv2.putText(output_image, pixel_text, (x+20, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    except Exception as e:
        print(f"Warning: Could not process output for frame {frame_number}: {e}")
        print(f"Raw output: {output}")
        # Draw a question mark in the center
        cv2.putText(output_image, "?", (width//2, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Create output filename
    output_filename = f"inference_{frame_number:06d}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save image
    cv2.imwrite(output_path, output_image)
    
    return output_path



def benchmark(model, images_folder, model_type, output_folder=None):
    """
    Run fast benchmark on image sequence.
    
    Args:
        model: Loaded model
        images_folder: Folder containing images
        model_type: Model type for display
        output_folder: Folder to save output images (1/10 of images)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
    
    if not image_files:
        print(f"Error: No images found in {images_folder}")
        return
    
    image_files.sort()  # Sort for consistent ordering
    print(f"Found {len(image_files)} images")
    print(f"Model: {model_type}")
    print(f"Device: {device}")
    
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")
    
    print("\nRunning benchmark...\n")
    
    # FPS tracking
    inference_times = []
    total_times = []
    frame_count = 0
    saved_frame_count = 0
    start_time = time.time()
    
    # For debugging: collect all raw outputs
    raw_outputs = []
    
    # Warmup run (first inference is usually slower)
    if image_files:
        warmup_image = cv2.imread(image_files[0])
        if warmup_image is not None:
            warmup_tensor = preprocess(warmup_image)
            with torch.no_grad():
                if TRT_AVAILABLE and isinstance(model, TRTModule):
                    output = model(warmup_tensor.half())
                else:
                    output = model(warmup_tensor)
                print(f"\nFirst inference raw output: {output.detach().cpu().numpy().flatten()}")
                print(f"Output shape: {output.shape}")
                print(f"Output dtype: {output.dtype}")
    
    # Benchmark loop
    try:
        for image_path in image_files:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Preprocess
            preprocess_start = time.time()
            image_tensor = preprocess(image)
            preprocess_time = time.time() - preprocess_start
            
            # Run inference
            inference_start = time.time()
            with torch.no_grad():
                if TRT_AVAILABLE and isinstance(model, TRTModule):
                    output_tensor = model(image_tensor.half())
                else:
                    output_tensor = model(image_tensor)
                
                # Get raw output as numpy array
                raw_output = output_tensor.detach().cpu().numpy().flatten()
                
            inference_end = time.time()
            
            # Calculate times
            inference_time = inference_end - inference_start
            total_time = inference_end - preprocess_start
            
            inference_times.append(inference_time)
            total_times.append(total_time)
            frame_count += 1
            
            # Store raw output for analysis
            raw_outputs.append(raw_output.copy())
            
            # Print first few raw outputs for debugging
            if frame_count <= 3:
                print(f"Frame {frame_count} raw output: {raw_output}")
            
            # Save 1/10 of the images with inference coordinates
            if output_folder and (frame_count % 10 == 1):
                try:
                    save_output_image(image, raw_output, frame_count, output_folder, raw_output)
                    saved_frame_count += 1
                except Exception as e:
                    print(f"Warning: Could not save image {frame_count}: {e}")
            
            # Progress update every 50 frames
            if frame_count % 50 == 0:
                current_avg_fps = 1.0 / np.mean(total_times[-50:]) if total_times else 0
                print(f"Processed {frame_count}/{len(image_files)} images - Current FPS: {current_avg_fps:.2f}", end='\r', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Calculate final statistics
    total_elapsed = time.time() - start_time
    
    if inference_times:
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Model Type: {model_type}")
        print(f"Total Images: {frame_count}")
        print(f"Total Time: {total_elapsed:.2f} seconds")
        if output_folder:
            print(f"Saved Output Images: {saved_frame_count}")
        
        # Print statistics about raw outputs
        if raw_outputs:
            raw_outputs_np = np.array(raw_outputs)
            print(f"\nModel Output Statistics:")
            for i in range(min(4, raw_outputs_np.shape[1])):  # Show first 4 outputs
                print(f"  Output[{i}]: mean={np.mean(raw_outputs_np[:, i]):.6f}, "
                      f"std={np.std(raw_outputs_np[:, i]):.6f}, "
                      f"min={np.min(raw_outputs_np[:, i]):.6f}, "
                      f"max={np.max(raw_outputs_np[:, i]):.6f}")
        
        print(f"\nInference FPS (inference only):")
        print(f"  Average: {1.0/np.mean(inference_times):.2f} FPS")
        print(f"  Min: {1.0/np.max(inference_times):.2f} FPS")
        print(f"  Max: {1.0/np.min(inference_times):.2f} FPS")
        print(f"  Std: {np.std([1.0/t for t in inference_times]):.2f} FPS")
        print(f"\nTotal FPS (preprocess + inference):")
        print(f"  Average: {1.0/np.mean(total_times):.2f} FPS")
        print(f"  Min: {1.0/np.max(total_times):.2f} FPS")
        print(f"  Max: {1.0/np.min(total_times):.2f} FPS")
        print(f"  Std: {np.std([1.0/t for t in total_times]):.2f} FPS")
        print(f"\nThroughput: {frame_count/total_elapsed:.2f} images/second")
        print("="*60)
        
        # Optionally save raw outputs to file
        if output_folder:
            raw_output_file = os.path.join(output_folder, "raw_outputs.csv")
            np.savetxt(raw_output_file, raw_outputs_np, delimiter=",", fmt="%.6f")
            print(f"Raw outputs saved to: {raw_output_file}")
    else:
        print("Error: No frames processed")


def main():
    parser = argparse.ArgumentParser(description='Fast benchmark for model inference on image sequence')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Folder containing images to process')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'resnet18', 'resnet50', 'wide_resnet50_2'],
                       help='Model architecture type (use "auto" to detect from checkpoint)')
    parser.add_argument('--use_trt', action='store_true',
                       help='Load TRT optimized model (requires torch2trt)')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Folder to save output images with inference coordinates (saves 1/10 of images)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found: {args.images_folder}")
        return
    
    # Load model
    model = load_model(args.model, args.model_type, args.use_trt)
    
    # Determine model type for display
    if args.model_type == 'auto':
        detected = detect_model_type_from_checkpoint(args.model)
        display_type = detected if detected else 'resnet18'
    else:
        display_type = args.model_type
    
    # Run benchmark
    benchmark(model, args.images_folder, display_type, args.output_folder)


if __name__ == '__main__':
    main()
