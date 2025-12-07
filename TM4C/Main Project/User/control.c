// control.c
// Mode decision logic (all in this file).
//  - MANUAL : RC control via manual.c
//  - AUTO   : Vision/data control via auto.c
//
// Rules:
//   1) At power-up, start in AUTO mode (ignore RC).
//   2) Periodically sample RC pulses:
//        * If we see ANY non-neutral command -> enter MANUAL.
//        * If we are in MANUAL and BOTH channels stay neutral
//          for >= 3 seconds -> go back to AUTO.
//   3) STOP mode is not used; AUTO should output neutral when it has no data.

#include "control.h"
#include "manual.h"
#include "auto.h"
#include "pwm_output.h"
#include "rc_input.h"

#define NEUTRAL_US    1500U
#define NEUTRAL_TOL   40U        // tolerance around neutral to treat as "neutral"

// One control frame ~20 ms.
// We sample RC every N frames: N * 20 ms = RC sample interval.
#define RC_CHECK_INTERVAL_FRAMES        5U    // 5 * 20ms ˜ 100 ms

// If both channels are neutral for this many RC samples in MANUAL,
// we treat it as "user stopped using RC" and go back to AUTO.
// 30 samples * 100 ms ˜ 3 seconds.
#define MANUAL_NEUTRAL_SAMPLES_THRESHOLD 30U

static control_mode_t g_curr_mode = CONTROL_MODE_STOP;

// 0: use AUTO (RC not active)
// 1: use MANUAL (RC active)
static uint8_t  g_manual_enabled        = 0U;
static uint32_t g_rc_check_counter      = 0U;
static uint32_t g_manual_neutral_samples = 0U;

// -----------------------------------------------------------
// Helper: absolute difference of two unsigned values
// -----------------------------------------------------------
static uint16_t abs_diff_u16(uint16_t a, uint16_t b)
{
    return (a > b) ? (a - b) : (b - a);
}

// -----------------------------------------------------------
// Helper: sample RC pulses and update g_manual_enabled.
//
//  - When MANUAL is OFF (g_manual_enabled == 0):
//        If any channel is non-neutral -> enter MANUAL.
//  - When MANUAL is ON (g_manual_enabled == 1):
//        If both channels stay neutral for >= 3 s -> back to AUTO.
// -----------------------------------------------------------
static void Control_UpdateModeByRcActivity(void)
{
    // Count control frames and only sample RC every RC_CHECK_INTERVAL_FRAMES
    g_rc_check_counter++;
    if (g_rc_check_counter < RC_CHECK_INTERVAL_FRAMES) {
        return;
    }
    g_rc_check_counter = 0U;

    // Sample RC pulses once (blocking, but only every ~100 ms)
    uint16_t thr = RC_GetThrottlePulseUs();
    uint16_t str = RC_GetSteerPulseUs();

    uint16_t diff_thr = abs_diff_u16(thr, NEUTRAL_US);
    uint16_t diff_str = abs_diff_u16(str, NEUTRAL_US);

    uint8_t is_thr_neutral = (diff_thr <= NEUTRAL_TOL);
    uint8_t is_str_neutral = (diff_str <= NEUTRAL_TOL);

    if (!g_manual_enabled) {
        // Currently in AUTO mode.
        // If any channel is non-neutral -> enable MANUAL.
        if (!is_thr_neutral || !is_str_neutral) {
            g_manual_enabled = 1U;
            g_manual_neutral_samples = 0U;  // reset counter when entering MANUAL
        }
    } else {
        // Currently in MANUAL mode.
        // If both channels are neutral -> count neutral samples.
        if (is_thr_neutral && is_str_neutral) {
            if (g_manual_neutral_samples < 100000U) {
                g_manual_neutral_samples++;
            }
        } else {
            // Any non-neutral command resets the neutral counter.
            g_manual_neutral_samples = 0U;
        }

        // If we stayed neutral for >= threshold -> go back to AUTO.
        if (g_manual_neutral_samples >= MANUAL_NEUTRAL_SAMPLES_THRESHOLD) {
            g_manual_enabled = 0U;          // back to AUTO
            g_manual_neutral_samples = 0U;  // reset counter
        }
    }
}

// -----------------------------------------------------------
// Public API
// -----------------------------------------------------------

void Control_Init(void)
{
    g_curr_mode               = CONTROL_MODE_STOP;
    g_manual_enabled          = 0U;    // start in AUTO mode
    g_rc_check_counter        = 0U;
    g_manual_neutral_samples  = 0U;
}

control_mode_t Control_GetMode(void)
{
    return g_curr_mode;
}

void Control_RunFrame(void)
{
    // 1) Update MANUAL/AUTO selection based on RC activity.
    Control_UpdateModeByRcActivity();

    // 2) Decide mode and run one frame.
    if (g_manual_enabled) {
        // MANUAL mode: RC control
        g_curr_mode = CONTROL_MODE_MANUAL;
        Manual_RunFrame();
    } else {
        // AUTO mode: vision/data control
        g_curr_mode = CONTROL_MODE_AUTO;
        Auto_RunFrame();
    }

    // NOTE:
    //  - STOP mode is not used here. If AUTO has no valid data,
    //    Auto_RunFrame() should output neutral (safe stop) itself.
}
