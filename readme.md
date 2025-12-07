## TM4C development guide
In order to compile the TM4C project properly, please do copy the lib folder under the TM4C folder.
During coding, one should seprate their code in a three layer style.

## JetsonNano development guide
fan control
```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' #Maximun Speed
```

### Connect to  the selected WiFi network

    >  For Mobile use

    ```bash
    sudo nmcli device wifi connect MaoLove password 39393939
    ```

	> For Dorm Wifi

	```bash
	sudo nmcli device wifi connect MUST-T1-4D password Must@28881122
	```

### release camera
```bash
sudo -S systemctl restart nvargus-daemon 
```