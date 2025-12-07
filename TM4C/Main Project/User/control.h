// control.h
// Mode manager: select MANUAL / AUTO / STOP.

#ifndef CONTROL_H_
#define CONTROL_H_

#include <stdint.h>

typedef enum {
    CONTROL_MODE_STOP = 0,
    CONTROL_MODE_MANUAL,
    CONTROL_MODE_AUTO
} control_mode_t;

// Init internal state (current mode, etc.)
void Control_Init(void);

// One control frame (~20 ms):
//  - Decide current mode (MANUAL / AUTO / STOP)
//  - Call Manual_RunFrame() / Auto_RunFrame() / neutral output
void Control_RunFrame(void);

// Optional: get current mode (for debug / LED / UART)
control_mode_t Control_GetMode(void);

#endif // CONTROL_H_
