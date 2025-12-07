// auto.h
// AUTO mode: convert speed + steering (from upper board) to PWM.

#ifndef AUTO_H_
#define AUTO_H_

#include <stdint.h>

// Init AUTO mode (and data simulator).
void Auto_Init(void);

// One AUTO frame (~20 ms):
//  - Get speed + steering command from data.c
//  - Map to ESC / servo pulse width
//  - Output PWM via pwm_output.c
void Auto_RunFrame(void);

#endif // AUTO_H_
