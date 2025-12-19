## TM4C development guide
In order to compile the TM4C project properly, please do copy the lib folder under the TM4C folder.
During coding, one should seprate their code in a three layer style.

## JetsonNano development guide
### Steering and Throttle
Currently,
+ Throttle 
	MAX: 800(Left) MIN-450(Right)
	CENTER: Tobe Define
### fan control
```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' #Maximun Speed
```

### Connect to  the selected WiFi network

    >  For Mobile use

    ```bash
    sudo nmcli device wifi connect Ohio password 1145141919810
    ```

	> For Dorm Wifi

	```bash
	sudo nmcli device wifi connect MUST-T1-4D password Must@28881122
	```

### release camera
```bash
sudo -S systemctl restart nvargus-daemon 
```