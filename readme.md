# Kaze Racing

## About Project
This project aims to design and implement an on-track autonomous driving car using the visual solution.

- **Desktop Training**  
  Used for model training and validation. Training is performed on a desktop environment with GPU acceleration to efficiently optimize the neural network.

- **Jetson Nano (On-board Inference)**  
  Deployed on the autonomous racing car for real-time visual inference. The trained model is transferred to the Jetson Nano, where it performs on-track perception and decision-making.

- **TM4C Microcontroller (Vehicle Firmware)**  
  Responsible for low-level vehicle control. The TM4C microcontroller runs the firmware that handles motor control, steering, and real-time communication with the Jetson Nano.

### 

## Requirements: software
- Operating System: Windows 10/ Windows 11 / Linux
- Python 3.10
- CUDA
- PyTorch
- OpenCV
- Conda
- Jetpack 4.6.1

## Environment Setup
```yaml
name: kzr
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch==2.8.0+cu128
      - torchvision==0.23.0+cu128
      - cupy-cuda13x==13.6.0
```

### Step 1: Install Conda
Make sure Conda (Anaconda or Miniconda) is installed on your system.

- Anaconda: https://www.anaconda.com/products/distribution  
- Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html

---

### Step 2: Create the Conda Environment

Clone the repository and create the environment using the provided `environment.yml` file:

```bash
git clone https://github.com/KessokuDrive/KazeRacing25.git
cd KazeRacing25
conda env create -f environment.yml
```

### Step 3: Activate the Environment

```bash
conda activate kzr
```

## Pretrained Models
**ResNet-18**

## Preparation for Testing


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
