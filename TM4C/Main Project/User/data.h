// data.h
// Simulated vision-board high level commands (speed + steering).

#ifndef DATA_H_
#define DATA_H_

#include <stdint.h>

// High level command from upper board
//  speed: -1000..+1000 (negative = reverse / brake, positive = forward)
//  steer: -1000..+1000 (negative = left, positive = right)
typedef struct {
    int16_t speed;   // normalized speed command
    int16_t steer;   // normalized steering command
    uint8_t valid;   // 1: command valid, 0: no command
} data_cmd_t;

void Data_Init(void);

// Called once per control frame (~20 ms).
void Data_RunFrame(void);

// Copy latest command to *cmd, return 1 if valid == 1, else 0.
uint8_t Data_GetCmd(data_cmd_t *cmd);

#endif // DATA_H_
