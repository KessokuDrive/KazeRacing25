// pwm_output.h
// Generate 50 Hz PWM on PE4 (ESC) and PE5 (servo).

#ifndef PWM_OUTPUT_H_
#define PWM_OUTPUT_H_

#include <stdint.h>

// Configure PE4 / PE5 as push-pull outputs, idle low.
void PWM_Output_Init(void);

// Blocking, one 20 ms frame:
//   esc_us   : ESC pulse width   (1000~2000 us)
//   servo_us : Servo pulse width (1000~2000 us)
void PWM_Output_WriteFrame(uint16_t esc_us, uint16_t servo_us);

#endif // PWM_OUTPUT_H_
