// rc_input.c
// RC pulse measurement using Timer0A/0B capture interrupts on TM4C123.
// PB6 -> T0CCP0 (steering), PB7 -> T0CCP1 (throttle).
//
// Timer0A/B are configured as 16-bit capture timers in edge-time mode,
// counting UP at F_CPU. We handle the possible single 16-bit wrap between
// rising and falling edges so that 1.0–2.0 ms pulses are measured correctly,
// even at 50 MHz.

#include "rc_input.h"
#include "tm4c123gh6pm.h"
#include <stdint.h>

#ifndef F_CPU
#define F_CPU 50000000UL   // System clock in Hz (adjust if different)
#endif

#define TICKS_PER_US   (F_CPU / 1000000UL)   // e.g. 50 -> 1 us = 50 ticks

// Expected valid pulse range (microseconds)
#define PULSE_MIN_US   900UL
#define PULSE_MAX_US   2100UL

// Convert to ticks at F_CPU
#define PULSE_MIN_TICKS   (PULSE_MIN_US * TICKS_PER_US)
#define PULSE_MAX_TICKS   (PULSE_MAX_US * TICKS_PER_US)

// Latest measured pulse widths (microseconds), default neutral = 1500 us
static volatile uint16_t g_thr_pulse_us = 1500U;   // PB7 / Timer0B
static volatile uint16_t g_str_pulse_us = 1500U;   // PB6 / Timer0A

// Last rising-edge timestamps (16-bit timer value) and edge state flags
static volatile uint16_t g_str_last_edge = 0U;
static volatile uint8_t  g_str_wait_fall = 0U;

static volatile uint16_t g_thr_last_edge = 0U;
static volatile uint8_t  g_thr_wait_fall = 0U;

// ---------- 16-bit unwrap helper ----------
//
// We know the true high-time is between PULSE_MIN_TICKS and PULSE_MAX_TICKS.
// With a 16-bit timer, there can be at most one wrap (0x0000 -> 0xFFFF).
//
// diff_mod = (now - start) in modulo-65536 arithmetic.
// If PULSE_MAX_TICKS = 65535, no wrap is possible.
// If PULSE_MAX_TICKS > 65535, some pulses will wrap exactly once.
//
// For the general case we reconstruct the true tick count from diff_mod.
static uint32_t unwrap_ticks(uint16_t diff_mod)
{
    const uint32_t wrap_limit = (PULSE_MAX_TICKS > 65536UL)
                                ? (PULSE_MAX_TICKS - 65536UL)
                                : 0UL;

    uint32_t diff = diff_mod;

    if (PULSE_MAX_TICKS <= 65535UL) {
        // No wrap case: simple 16-bit difference is enough.
        return diff;
    }

    if (diff <= wrap_limit) {
        // Wrapped once: add one full 16-bit period.
        diff += 65536UL;
    } else if (diff < PULSE_MIN_TICKS) {
        // Outside any plausible region: treat as invalid.
        return 0UL;
    }

    return diff;
}

// Convert tick count to microseconds and check range.
// Returns 0 for invalid measurements.
static uint16_t ticks_to_us_checked(uint32_t ticks)
{
    if (ticks == 0UL) {
        return 0U;
    }

    uint32_t us = ticks / TICKS_PER_US;

    if (us < PULSE_MIN_US || us > PULSE_MAX_US) {
        return 0U;   // out of RC range
    }

    return (uint16_t)us;
}

// ---------------- Public API ----------------

void RC_Input_Init(void)
{
    // 1) Enable GPIOB and Timer0 clocks
    SYSCTL_RCGCGPIO_R  |= SYSCTL_RCGCGPIO_R1;   // GPIOB
    SYSCTL_RCGCTIMER_R |= SYSCTL_RCGCTIMER_R0;  // TIMER0
    (void)SYSCTL_RCGCGPIO_R;                    // dummy reads
    (void)SYSCTL_RCGCTIMER_R;

    // 2) Configure PB6 / PB7 as T0CCP0 / T0CCP1 capture pins

    GPIO_PORTB_DIR_R   &= ~((1U << 6) | (1U << 7));  // input
    GPIO_PORTB_DEN_R   |=  ((1U << 6) | (1U << 7));  // digital enable
    GPIO_PORTB_AFSEL_R |=  ((1U << 6) | (1U << 7));  // alternate function
    GPIO_PORTB_PDR_R   |=  ((1U << 6) | (1U << 7));  // weak pull-down
    GPIO_PORTB_AMSEL_R &= ~((1U << 6) | (1U << 7));  // disable analog

    // PB6 = T0CCP0, PB7 = T0CCP1 (function 7)
    GPIO_PORTB_PCTL_R &= ~((0xFU << 24) | (0xFU << 28));
    GPIO_PORTB_PCTL_R |=  ((0x7U << 24) | (0x7U << 28));

    // 3) Configure Timer0A/B as 16-bit capture, edge-time, count-up

    // Disable during configuration
    TIMER0_CTL_R &= ~(TIMER_CTL_TAEN | TIMER_CTL_TBEN);

    // 16-bit mode for both A and B
    TIMER0_CFG_R = TIMER_CFG_16_BIT;

    // ---- Timer0A: PB6 (steering) ----
    TIMER0_TAMR_R = TIMER_TAMR_TAMR_CAP   |   // capture mode
                    TIMER_TAMR_TACMR      |   // edge-time mode
                    TIMER_TAMR_TACDIR;        // count up
    TIMER0_TAPR_R  = 0U;                      // no prescaler
    TIMER0_TAILR_R = 0xFFFFU;

    // Capture rising edge first (TAEVENT = 0)
    TIMER0_CTL_R &= ~TIMER_CTL_TAEVENT_M;

    // Enable capture interrupt on Timer0A
    TIMER0_IMR_R |= TIMER_IMR_CAEIM;

    // ---- Timer0B: PB7 (throttle) ----
    TIMER0_TBMR_R = TIMER_TBMR_TBMR_CAP   |
                    TIMER_TBMR_TBCMR      |
                    TIMER_TBMR_TBCDIR;        // count up
    TIMER0_TBPR_R  = 0U;
    TIMER0_TBILR_R = 0xFFFFU;

    // Capture rising edge first
    TIMER0_CTL_R &= ~TIMER_CTL_TBEVENT_M;

    // Enable capture interrupt on Timer0B
    TIMER0_IMR_R |= TIMER_IMR_CBEIM;

    // 4) Enable Timer0A/B interrupts in NVIC (vectors 19, 20)
    NVIC_EN0_R |= (1U << 19) | (1U << 20);

    // 5) Start timers
    TIMER0_CTL_R |= (TIMER_CTL_TAEN | TIMER_CTL_TBEN);
}

uint16_t RC_GetThrottlePulseUs(void)
{
    return g_thr_pulse_us;
}

uint16_t RC_GetSteerPulseUs(void)
{
    return g_str_pulse_us;
}

// ---------------- ISRs ----------------

// Timer0A capture ISR: PB6 (steering)
void TIMER0A_Handler(void)
{
    uint16_t now = (uint16_t)TIMER0_TAR_R;

    if (!g_str_wait_fall) {
        // Rising edge: store start time, wait for falling edge next
        g_str_last_edge = now;
        g_str_wait_fall = 1U;

        TIMER0_CTL_R &= ~TIMER_CTL_TAEVENT_M;
        TIMER0_CTL_R |=  TIMER_CTL_TAEVENT_NEG;   // falling edge
    } else {
        // Falling edge: compute modulo-65536 diff
        uint16_t diff_mod   = (uint16_t)(now - g_str_last_edge);
        uint32_t true_ticks = unwrap_ticks(diff_mod);
        uint16_t us         = ticks_to_us_checked(true_ticks);

        if (us != 0U) {
            g_str_pulse_us = us;
        }

        // Back to rising edge
        g_str_wait_fall = 0U;
        TIMER0_CTL_R   &= ~TIMER_CTL_TAEVENT_M;
    }

    // Clear Timer0A capture event interrupt
    TIMER0_ICR_R = TIMER_ICR_CAECINT;
}

// Timer0B capture ISR: PB7 (throttle)
void TIMER0B_Handler(void)
{
    uint16_t now = (uint16_t)TIMER0_TBR_R;

    if (!g_thr_wait_fall) {
        // Rising edge
        g_thr_last_edge = now;
        g_thr_wait_fall = 1U;

        TIMER0_CTL_R &= ~TIMER_CTL_TBEVENT_M;
        TIMER0_CTL_R |=  TIMER_CTL_TBEVENT_NEG;   // falling edge
    } else {
        uint16_t diff_mod   = (uint16_t)(now - g_thr_last_edge);
        uint32_t true_ticks = unwrap_ticks(diff_mod);
        uint16_t us         = ticks_to_us_checked(true_ticks);

        if (us != 0U) {
            g_thr_pulse_us = us;
        }

        g_thr_wait_fall = 0U;
        TIMER0_CTL_R   &= ~TIMER_CTL_TBEVENT_M;
    }

    // Clear Timer0B capture event interrupt
    TIMER0_ICR_R = TIMER_ICR_CBECINT;
}
