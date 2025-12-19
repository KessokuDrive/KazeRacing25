// manual.h
// Manual RC passthrough (F/R ESC + steering servo).

#ifndef MANUAL_H_
#define MANUAL_H_

#include <stdint.h>

// Init internal state (neutral, counters).
void Manual_Init(void);

// One manual frame (~20 ms):
//  - Read RC pulses (PB7 throttle, PB6 steering)
//  - Apply deadband / simple checks
//  - Output one PWM frame on PE4/PE5
void Manual_RunFrame(void);

// Configure global RC adjustments:
//  scale_pct: percentage applied to diff from neutral (100 = no change)
//  bias_us  : microsecond offset added after scaling
void Manual_SetRcAdjust(int16_t thr_scale_pct, int16_t str_scale_pct,
                        int16_t thr_bias_us,   int16_t str_bias_us);

#endif // MANUAL_H_
