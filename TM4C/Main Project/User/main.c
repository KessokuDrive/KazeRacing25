// main.c
// TM4C123 car controller
//  - RC input   : PB7 (throttle), PB6 (steering)
//  - PWM output : PE4 (ESC), PE5 (servo)
//  - Modes      : MANUAL / AUTO / STOP (selected in control.c)

#include <stdint.h>
#include "tm4c123gh6pm.h"

#include "rc_input.h"
#include "pwm_output.h"
#include "manual.h"
#include "auto.h"
#include "control.h"

#ifndef F_CPU
#define F_CPU 50000000UL   // 50 MHz system clock (assumed)
#endif

int main(void)
{
    uint32_t i;

    // ---- Peripheral / module init ----
    // System clock is assumed configured by startup code.
    RC_Input_Init();       // Configure PB6 / PB7 capture for RC
    PWM_Output_Init();     // Configure PE4 / PE5 PWM output (50 Hz)
    Manual_Init();         // Manual (RC) mode state
    Auto_Init();           // now uses real UART vision data
    Control_Init();        // Mode manager (start in STOP)

    // Enable global interrupts (used by rc_input if using timer capture)
    __enable_irq();

    // ---- Neutral pulses to arm ESC / servo (~2 s) ----
    // Many ESCs need some neutral pulses after power-on.
    for (i = 0; i < 100U; i++) {       // 100 * 20 ms  2 s
        PWM_Output_WriteFrame(1500U, 1500U);   // ESC neutral, servo center
    }

    // ---- Main control loop ----
    while (1) {
        // One control frame:
        //  - Control_RunFrame() decides mode:
        //      MANUAL > AUTO > STOP
        //  - Calls Manual_RunFrame() / Auto_RunFrame()
        //    or outputs neutral in STOP mode.
        //  - Each RunFrame internally generates one ~20 ms PWM frame.
        Control_RunFrame();
    }
}
