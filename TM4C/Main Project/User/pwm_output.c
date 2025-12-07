// pwm_output.c
// Simple 50 Hz software PWM on PE4 (ESC) and PE5 (servo).
// Uses SysTick (NVIC_ST_* registers) as a microsecond delay source.

#include "pwm_output.h"
#include "tm4c123gh6pm.h"
#include <stdint.h>

#ifndef F_CPU
#define F_CPU 50000000UL
#endif

#define FRAME_US   20000U    // 20 ms frame
#define GAP_US       300U    // gap between ESC and servo pulses

/* -------- SysTick microsecond delay (blocking) -------- */
static void delay_us(uint32_t us)
{
    uint32_t ticks;

    if (us == 0U) return;

    // Handle long delays by chunks (SysTick is 24-bit)
    while (us > 0U) {
        uint32_t chunk = us;
        if (chunk > 1000U) chunk = 1000U;      // max 1000 us per chunk

        ticks = (F_CPU / 1000000UL) * chunk;
        if (ticks > 0x00FFFFFFUL) {
            ticks = 0x00FFFFFFUL;
        }

        NVIC_ST_CTRL_R = 0;                   // disable
        NVIC_ST_RELOAD_R = ticks - 1U;        // reload value
        NVIC_ST_CURRENT_R = 0;                // clear current
        // enable, core clock, no interrupt
        NVIC_ST_CTRL_R = NVIC_ST_CTRL_ENABLE | NVIC_ST_CTRL_CLK_SRC;

        // Wait for COUNTFLAG
        while ((NVIC_ST_CTRL_R & NVIC_ST_CTRL_COUNT) == 0U) {
            // busy wait
        }

        NVIC_ST_CTRL_R = 0;                   // stop
        us -= chunk;
    }
}

void PWM_Output_Init(void)
{
    // Enable Port E clock
    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R4;   // GPIOE
    (void)SYSCTL_RCGCGPIO_R;

    // PE4 / PE5 as digital outputs, idle low
    GPIO_PORTE_DIR_R   |=  (1U << 4) | (1U << 5);
    GPIO_PORTE_DEN_R   |=  (1U << 4) | (1U << 5);
    GPIO_PORTE_AFSEL_R &= ~((1U << 4) | (1U << 5));
    GPIO_PORTE_DR2R_R  |=  (1U << 4) | (1U << 5); // 2mA drive
    GPIO_PORTE_DATA_R  &= ~((1U << 4) | (1U << 5));
}

void PWM_Output_WriteFrame(uint16_t esc_us, uint16_t servo_us)
{
    uint32_t used = 0U;

    // Basic clamp for safety
    if (esc_us   < 1000U) esc_us   = 1000U;
    if (esc_us   > 2000U) esc_us   = 2000U;
    if (servo_us < 1000U) servo_us = 1000U;
    if (servo_us > 2000U) servo_us = 2000U;

    // ---- ESC pulse on PE4 ----
    GPIO_PORTE_DATA_R |=  (1U << 4);
    delay_us(esc_us);
    GPIO_PORTE_DATA_R &= ~(1U << 4);
    used += esc_us;

    // ---- small gap ----
    delay_us(GAP_US);
    used += GAP_US;

    // ---- Servo pulse on PE5 ----
    GPIO_PORTE_DATA_R |=  (1U << 5);
    delay_us(servo_us);
    GPIO_PORTE_DATA_R &= ~(1U << 5);
    used += servo_us;

    // ---- Wait remaining time of the 20 ms frame ----
    if (used < FRAME_US) {
        delay_us(FRAME_US - used);
    }
}
