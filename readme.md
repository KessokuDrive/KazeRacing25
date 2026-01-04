# Kaze Racing

## About Project
This project aims to design and implement an on-track autonomous driving car using the visual solution.\

We used **ResNet18** as base model, and trained an end-to-end mapping from camera captured image to steering and throttling signal.

Our project consist of the following three part.

- **Desktop Training**  
  Used for model training and validation. Training is performed on a desktop environment with GPU acceleration to efficiently optimize the neural network.

- **Jetson Nano (On-board Inference)**  
  Deployed on the autonomous racing car for real-time visual inference. The trained model is transferred to the Jetson Nano, where it performs on-track perception and decision-making.

- **TM4C Microcontroller (Vehicle Firmware)**  
  Responsible for low-level vehicle control. The TM4C microcontroller runs the firmware that handles motor control, steering, and real-time communication with the Jetson Nano.

# Install
## Traning(Desktop)
### Requirements
- Operating System: Windows 10/ Windows 11 / Linux
- Python 3.10
- CUDA
- PyTorch
- OpenCV
- Conda

### Step 1: Install Required Software
Make sure Conda (Anaconda or Miniconda) is installed on your system.
- Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html

---

### Step 2: Create the Conda Environment

Clone the repository:
```bash
git clone https://github.com/KessokuDrive/KazeRacing25.git
cd KazeRacing25
```
Create and the enviroment
```bash
conda env create -f desktop_requirement.yml
conda activate kzr
```

### Installation complete
Now, you may want to run the code for traning at `/DesktopTraning/train.py` or benchmark at `DesktopTraning/benchmark/benchmark.py`

## Inference (Jetson Nano)
### Install Jetpack
Find the mirror and guid at Nvidia:
+ Jetpack 4.6.1: https://developer.nvidia.com/embedded/jetpack-sdk-461

### Ready to Rock!
to start, type run this at jetson
```bash
sudo /usr/bin/python3 /home/jetson/jetracer/roadfollowing.py
```

# Note for devs
## TM4C Development Guide
In order to compile the TM4C project properly, please do copy the lib folder under the TM4C folder.
During coding, one should seprate their code in a three layer style.

## JetsonNano Development Guide
fan control
```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' #Maximun Speed
```

### Connect to the Selected WiFi Network

    >  For Mobile use

    ```bash
    sudo nmcli device wifi connect MaoLove password 39393939
    ```

	> For Dorm Wifi

	```bash
	sudo nmcli device wifi connect MUST-T1-4D password Must@28881122
	```

### Release Camera
```bash
sudo -S systemctl restart nvargus-daemon 
```
