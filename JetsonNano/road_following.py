import serial
import torch
from torch2trt import TRTModule
from jetcam.csi_camera import CSICamera
from utils import preprocess
import numpy as np
import threading

MODELTRT = "kazeRacing003_trt.pth"

# Initialize UART
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
print("serial all right")
# Initialize camera
camera = CSICamera(width=224, height=224, capture_fps=65)
print("camera all right")
# Load optimized TRT model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('./model_trt/'+MODELTRT))

# Adjustable parameters
THROTTLE_MAX  = 200		# FULL speed
THROTTLE_MIN  = 140		# MIN  speed
THROTTLE_GAIN = 1	# Throttle sensitivity
THROTTLE_BIAS = 0.00	# Throttle offset

STEERING_MAXL = 850		#Left Most
STEERING_MAXR = -500	#Right Most
STEERING_GAIN = 1.2	# Steering sensitivity
STEERING_BIAS = 0.00	# Steering offset

# Autocalculate, do not change
steer_wing = (STEERING_MAXL - STEERING_MAXR)/2
steer_cent = STEERING_MAXL - steer_wing
throt_wing = (THROTTLE_MAX-THROTTLE_MIN)/2
throt_cent = THROTTLE_MIN + throt_wing

# Program state control
class ProgramState:
    def __init__(self):
        self.running = True
        self.paused = False
        self.lock = threading.Lock()
        
    def pause(self):
        with self.lock:
            self.paused = True
            
    def resume(self):
        with self.lock:
            self.paused = False
            
    def is_paused(self):
        with self.lock:
            return self.paused
            
    def stop(self):
        with self.lock:
            self.running = False
            
    def is_running(self):
        with self.lock:
            return self.running

state = ProgramState()

def send_command(ser, throttle, steering):
    """Send a command to the TM4C via UART"""
    command = f"T:{throttle},S:{steering}\n"
    ser.write(command.encode())

def main():
    try:
        print("Press 'p' to pause and configure parameters")
        
        while state.is_running():
            # Skip processing if paused
            if state.is_paused():
                continue
                
            # Get camera frame and process
            image = camera.read()
            image = preprocess(image).half()
            
            # Run inference
            output = model_trt(image).detach().cpu().numpy().flatten()
            x = float(output[0])
            y = float(output[1])
            
            # Calculate steering with adjustments
            steering = (x * STEERING_GAIN)*steer_wing + steer_cent
            steering = max(STEERING_MAXR, min(STEERING_MAXL, steering))  # Clamp
            print(x, steering)
            
            # Calculate throttle with adjustments
            throttle = 250
            #throttle = y*THROTTLE_GAIN * throt_wing + throt_cent
            #throttle = max(THROTTLE_MIN, min(THROTTLE_MAX, throttle))
            
            # Send control values via UART
            send_command(ser, throttle, steering)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        state.stop()
        ser.close()

if __name__ == "__main__":
    main()
