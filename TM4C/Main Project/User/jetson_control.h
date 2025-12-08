// jetson_control.h
// UART1 interface to receive throttle/steering commands from Jetson Nano.
// Uses UART1: PB0 (U1RX), PB1 (U1TX)

#ifndef JETSON_CONTROL_H_
#define JETSON_CONTROL_H_

#include <stdint.h>
#include "data.h"

// Initialize UART1 for Jetson communication (115200 baud, 8N1)
void UART_Vision_Init(void);

// Get latest command from Jetson
// Returns 1 if valid command received, 0 otherwise
uint8_t Jetson_GetCmd(data_cmd_t *cmd);

// Check if new data available (called periodically)
void Jetson_ProcessRx(void);

#endif // JETSON_CONTROL_H_

