import serial
import torch
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from utils import preprocess
import numpy as np

MODELTRT = "kazeRacing001_trt.pth"

# Initialize UART
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)

# Initialize camera
camera = CSICamera(width=224, height=224, capture_fps=65)

# Load optimized TRT model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('./model_trt/'+MODELTRT))

# Adjustable parameters
THROTTLE_GAIN = 0.15  # Base throttle value
STEERING_GAIN = 0.75  # Steering sensitivity
STEERING_BIAS = 0.00  # Steering offset
THROTTLE_ADJUST = 0.0  # Throttle adjustment (-1.0 to 1.0)
STEERING_ADJUST = 0.0  # Steering adjustment (-1.0 to 1.0)

def send_uart(throttle, steering):
    # Convert values to bytes and send via UART
    data = f"{throttle:.2f},{steering:.2f}\n".encode()
    ser.write(data)

def main():
    try:
        while True:
            # Get camera frame and process
            image = camera.read()
            image = preprocess(image).half()
            
            # Run inference
            output = model_trt(image).detach().cpu().numpy().flatten()
            x = float(output[0])
            
            # Calculate steering with adjustments
            steering = x * STEERING_GAIN + STEERING_BIAS + STEERING_ADJUST
            steering = max(-1.0, min(1.0, steering))  # Clamp to [-1, 1]
            
            # Calculate throttle with adjustments
            throttle = THROTTLE_GAIN + THROTTLE_ADJUST
            throttle = max(0.0, min(1.0, throttle))  # Clamp to [0, 1]
            
            # Send control values via UART
            send_uart(throttle, steering)
            print("throttle: {}, steering: {}".format(throttle, steering))
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
