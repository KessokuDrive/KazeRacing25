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
static volatile uint16_t g_mode_pulse_us = 2000U;  // PB2 / Timer3A (default 2ms = MANUAL)

// Last rising-edge timestamps (16-bit timer value) and edge state flags
static volatile uint16_t g_str_last_edge = 0U;
static volatile uint8_t  g_str_wait_fall = 0U;

static volatile uint16_t g_thr_last_edge = 0U;
static volatile uint8_t  g_thr_wait_fall = 0U;

static volatile uint16_t g_mode_last_edge = 0U;
static volatile uint8_t  g_mode_wait_fall = 0U;

// Mode switch signal timeout tracking (frame-based, not SysTick)
// Each frame is ~20ms, so 5 frames = 100ms timeout
static volatile uint32_t g_mode_pulse_count = 0U;  // increments each valid pulse

// Mode switch thresholds (in microseconds)
#define MODE_AUTO_PULSE_US     900U    // 0.9ms -> AUTO mode
#define MODE_MANUAL_PULSE_US   2000U   // 2.0ms -> MANUAL mode
#define MODE_THRESHOLD_US      1450U   // midpoint between 900 and 2000
#define MODE_TIMEOUT_FRAMES    5U      // if no new pulse for 5 frames (~100ms), default to MANUAL

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
    // 1) Enable GPIOB and Timer0/Timer3 clocks
    SYSCTL_RCGCGPIO_R  |= SYSCTL_RCGCGPIO_R1;   // GPIOB
    SYSCTL_RCGCTIMER_R |= SYSCTL_RCGCTIMER_R0;  // TIMER0
    SYSCTL_RCGCTIMER_R |= SYSCTL_RCGCTIMER_R3;  // TIMER3 for PB2
    (void)SYSCTL_RCGCGPIO_R;                    // dummy reads
    (void)SYSCTL_RCGCTIMER_R;

    // 2) Configure PB2, PB6, PB7 as capture pins

    GPIO_PORTB_DIR_R   &= ~((1U << 2) | (1U << 6) | (1U << 7));  // input
    GPIO_PORTB_DEN_R   |=  ((1U << 2) | (1U << 6) | (1U << 7));  // digital enable
    GPIO_PORTB_AFSEL_R |=  ((1U << 2) | (1U << 6) | (1U << 7));  // alternate function
    GPIO_PORTB_PDR_R   |=  ((1U << 2) | (1U << 6) | (1U << 7));  // weak pull-down
    GPIO_PORTB_AMSEL_R &= ~((1U << 2) | (1U << 6) | (1U << 7));  // disable analog

    // PB2 = T3CCP0 (function 7), PB6 = T0CCP0, PB7 = T0CCP1 (function 7)
    GPIO_PORTB_PCTL_R &= ~((0xFU << 8) | (0xFU << 24) | (0xFU << 28));
    GPIO_PORTB_PCTL_R |=  ((0x7U << 8) | (0x7U << 24) | (0x7U << 28));

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

    // 5) Configure Timer3A for PB2 (mode switch) - similar to Timer0A
    TIMER3_CTL_R &= ~TIMER_CTL_TAEN;              // disable during config
    
    TIMER3_CFG_R = TIMER_CFG_16_BIT;              // 16-bit mode
    
    TIMER3_TAMR_R = TIMER_TAMR_TAMR_CAP   |       // capture mode
                    TIMER_TAMR_TACMR      |       // edge-time mode
                    TIMER_TAMR_TACDIR;            // count up
    TIMER3_TAPR_R  = 0U;                          // no prescaler
    TIMER3_TAILR_R = 0xFFFFU;
    
    // Capture rising edge first
    TIMER3_CTL_R &= ~TIMER_CTL_TAEVENT_M;
    
    // Enable capture interrupt on Timer3A
    TIMER3_IMR_R |= TIMER_IMR_CAEIM;
    
    // Enable Timer3A interrupt in NVIC (vector 35)
    NVIC_EN1_R |= (1U << (35 - 32));
    
    // 6) Start timers
    TIMER0_CTL_R |= (TIMER_CTL_TAEN | TIMER_CTL_TBEN);
    TIMER3_CTL_R |= TIMER_CTL_TAEN;
    
    // NOTE: SysTick is NOT used here because pwm_output.c uses it for blocking delays.
    // Timeout detection uses frame-based counting instead.
}

uint16_t RC_GetThrottlePulseUs(void)
{
    return g_thr_pulse_us;
}

uint16_t RC_GetSteerPulseUs(void)
{
    return g_str_pulse_us;
}

uint16_t RC_GetModeSwitchPulseUs(void)
{
    return g_mode_pulse_us;
}

// Returns 1 for MANUAL mode, 0 for AUTO mode
// Default to MANUAL if no signal or timeout
uint8_t RC_IsManualMode(void)
{
    static uint32_t s_last_count = 0U;
    static uint8_t s_no_signal_frames = 0U;
    
    uint32_t current_count = g_mode_pulse_count;
    
    // Check if we received new pulses since last check
    if (current_count != s_last_count) {
        // Got new pulse(s), reset timeout counter
        s_no_signal_frames = 0U;
        s_last_count = current_count;
    } else {
        // No new pulse, increment timeout counter (capped to avoid overflow)
        if (s_no_signal_frames < 255U) {
            s_no_signal_frames++;
        }
    }
    
    // If no signal for MODE_TIMEOUT_FRAMES, default to MANUAL (fail-safe)
    if (s_no_signal_frames >= MODE_TIMEOUT_FRAMES) {
        return 1U;  // MANUAL mode (fail-safe)
    }
    
    // If pulse < threshold (closer to 900us), it's AUTO mode
    // If pulse >= threshold (closer to 2000us), it's MANUAL mode
    if (g_mode_pulse_us < MODE_THRESHOLD_US) {
        return 0U;  // AUTO mode
    } else {
        return 1U;  // MANUAL mode
    }
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

// Timer3A capture ISR: PB2 (mode switch)
void TIMER3A_Handler(void)
{
    uint16_t now = (uint16_t)TIMER3_TAR_R;
    
    if (!g_mode_wait_fall) {
        // Rising edge: store start time, wait for falling edge next
        g_mode_last_edge = now;
        g_mode_wait_fall = 1U;
        
        TIMER3_CTL_R &= ~TIMER_CTL_TAEVENT_M;
        TIMER3_CTL_R |=  TIMER_CTL_TAEVENT_NEG;   // falling edge
    } else {
        // Falling edge: compute modulo-65536 diff
        uint16_t diff_mod   = (uint16_t)(now - g_mode_last_edge);
        uint32_t true_ticks = unwrap_ticks(diff_mod);
        uint16_t us         = ticks_to_us_checked(true_ticks);
        
        if (us != 0U) {
            g_mode_pulse_us = us;
            g_mode_pulse_count++;  // increment pulse counter for timeout detection
        }
        
        // Back to rising edge
        g_mode_wait_fall = 0U;
        TIMER3_CTL_R   &= ~TIMER_CTL_TAEVENT_M;
    }
    
    // Clear Timer3A capture event interrupt
    TIMER3_ICR_R = TIMER_ICR_CAECINT;
}

// NOTE: SysTick_Handler removed - pwm_output.c uses SysTick for delays
// Timeout detection now uses frame-based pulse counting instead.
