// manual.c
// Manual mode: RC passthrough for F/R ESC and steering servo.
//  - Throttle: PB7  -> PE4 (ESC)
//  - Steering: PB6  -> PE5 (servo)

#include "manual.h"
#include "rc_input.h"
#include "pwm_output.h"

#ifndef F_CPU
#define F_CPU 50000000UL
#endif

#define NEUTRAL_US            1500U

#define THROTTLE_DEADBAND_US    40U   // +/-40 us -> neutral
#define SERVO_DEADBAND_US       20U   // +/-20 us -> neutral

#define PULSE_MIN_US          1000U
#define PULSE_MAX_US          2000U

// Global RC adjustments (scale in percent, bias in microseconds)
// Defaults: scale 100% (no change), bias 0 us.
static int16_t g_thr_bias_us = 0;
static int16_t g_str_bias_us = 0;
static int16_t g_thr_scale_pct = 50;
static int16_t g_str_scale_pct = 100;

// How many frames between RC updates (1 = every frame)
#define RC_UPDATE_EVERY_FRAMES   1U

static uint16_t g_esc_us   = NEUTRAL_US;  // ESC output pulse
static uint16_t g_servo_us = NEUTRAL_US;  // Servo output pulse
static uint32_t g_frame_cnt = 0U;

// ---- Local helpers -------------------------------------------------

static uint16_t clamp_us(uint16_t v)
{
    if (v < PULSE_MIN_US) return PULSE_MIN_US;
    if (v > PULSE_MAX_US) return PULSE_MAX_US;
    return v;
}

// Apply deadband around NEUTRAL_US
static uint16_t apply_deadband(uint16_t pulse, uint16_t db)
{
    int16_t diff = (int16_t)pulse - (int16_t)NEUTRAL_US;

    if ((diff > -(int16_t)db) && (diff < (int16_t)db)) {
        return NEUTRAL_US;
    } else {
        return pulse;
    }
}

// Apply global scale (% of diff from neutral) and bias (us) then clamp
static uint16_t apply_scale_and_bias(uint16_t pulse,
                                     int16_t scale_pct,
                                     int16_t bias_us)
{
    int32_t diff = (int32_t)pulse - (int32_t)NEUTRAL_US;
    diff = (diff * scale_pct) / 100;  // scale around neutral
    diff += bias_us;                  // add bias

    int32_t adjusted = (int32_t)NEUTRAL_US + diff;
    if (adjusted < (int32_t)PULSE_MIN_US) adjusted = PULSE_MIN_US;
    if (adjusted > (int32_t)PULSE_MAX_US) adjusted = PULSE_MAX_US;
    return (uint16_t)adjusted;
}

// Read RC once and update g_esc_us / g_servo_us
static void Manual_UpdateFromRC(void)
{
    // 1) Read RC pulses from PB7 / PB6
    uint16_t thr = RC_GetThrottlePulseUs(); // PB7
    uint16_t str = RC_GetSteerPulseUs();    // PB6

    // 2) Basic sanity check / clamp
    thr = clamp_us(thr);
    str = clamp_us(str);

    // 3) Deadband
    thr = apply_deadband(thr, THROTTLE_DEADBAND_US);
    str = apply_deadband(str, SERVO_DEADBAND_US);

    // 4) Apply global scaling/bias and clamp again
    thr = apply_scale_and_bias(thr, g_thr_scale_pct, g_thr_bias_us);
    str = apply_scale_and_bias(str, g_str_scale_pct, g_str_bias_us);

    // 4) Save to global output pulses
    g_esc_us   = thr;
    g_servo_us = str;
}

// ---- Public API ----------------------------------------------------

void Manual_Init(void)
{
    g_esc_us    = NEUTRAL_US;
    g_servo_us  = NEUTRAL_US;
    g_frame_cnt = 0U;
    // Keep existing bias/scale values; caller may set before Init.
}

void Manual_RunFrame(void)
{
    // Update RC reading periodically (here: every frame)
    if ((g_frame_cnt % RC_UPDATE_EVERY_FRAMES) == 0U) {
        Manual_UpdateFromRC();
    }

    g_frame_cnt++;

    // Output one ~20 ms frame at 50 Hz
    PWM_Output_WriteFrame(g_esc_us, g_servo_us);
}

// Set global RC scaling (percent) and bias (microseconds) for throttle/steer.
// Example: scale 110 increases response by 10%; bias 20 shifts center +20 us.
void Manual_SetRcAdjust(int16_t thr_scale_pct, int16_t str_scale_pct,
                        int16_t thr_bias_us,   int16_t str_bias_us)
{
    g_thr_scale_pct = thr_scale_pct;
    g_str_scale_pct = str_scale_pct;
    g_thr_bias_us   = thr_bias_us;
    g_str_bias_us   = str_bias_us;
}
