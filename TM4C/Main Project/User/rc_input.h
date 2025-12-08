// rc_input.h
// Read RC receiver pulses on PB7 (throttle) and PB6 (steering)
// using Timer0A/0B capture interrupts.

#ifndef RC_INPUT_H_
#define RC_INPUT_H_

#include <stdint.h>

// Initialize RC input module.
// PB6 -> T0CCP0 (Timer0A capture, steering)
// PB7 -> T0CCP1 (Timer0B capture, throttle)
void RC_Input_Init(void);

// Get latest measured throttle pulse width (in microseconds).
// Typical RC range: 1000–2000 us. Default (no valid pulse yet) is 1500 us.
uint16_t RC_GetThrottlePulseUs(void);

// Get latest measured steering pulse width (in microseconds).
// Typical RC range: 1000–2000 us. Default (no valid pulse yet) is 1500 us.
uint16_t RC_GetSteerPulseUs(void);

// Get latest measured mode switch pulse width (in microseconds).
// PB2 -> T3CCP0 (Timer3A capture, mode switch)
// 0.9ms (900us) -> AUTO mode, 2.0ms (2000us) -> MANUAL mode
uint16_t RC_GetModeSwitchPulseUs(void);

// Returns 1 for MANUAL mode, 0 for AUTO mode.
// Default to MANUAL if no signal or timeout (fail-safe).
// Mode threshold: < 1450us = AUTO, >= 1450us = MANUAL
uint8_t RC_IsManualMode(void);

#endif // RC_INPUT_H_
