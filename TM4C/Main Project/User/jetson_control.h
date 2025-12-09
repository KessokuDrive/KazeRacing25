// jetson_control.h
// UART0 communication with Jetson Nano for autonomous control
// Receives throttle and steering commands in the format:
//   "T:####,S:####\n" where #### are signed integers (-1000 to +1000)
// Example: "T:500,S:-200\n" means throttle=500, steer=-200

#ifndef JETSON_CONTROL_H_
#define JETSON_CONTROL_H_

#include <stdint.h>
#include "data.h"

// Initialize UART0 for Jetson communication
// UART0: RX=PA0, TX=PA1, 115200 baud, 8N1
void UART_Vision_Init(void);

// Process received UART data (call periodically in control loop)
// Reads incoming bytes and parses commands
void Jetson_ProcessRx(void);

// Get the latest valid command from Jetson
// Returns 1 if a valid command is available, 0 otherwise
// Populates *cmd with the latest throttle/steer values
uint8_t Jetson_GetCmd(data_cmd_t *cmd);

#endif // JETSON_CONTROL_H_
