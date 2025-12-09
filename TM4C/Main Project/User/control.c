// control.c
// Mode decision logic (all in this file).
//  - MANUAL : RC control via manual.c
//  - AUTO   : Vision/data control via auto.c
//
// Rules:
//   1) At power-up, start in MANUAL mode (RC control) - fail-safe default.
//   2) Mode is controlled by RC mode switch on PB2:
//        * High 2ms (or no signal) -> MANUAL mode (fail-safe)
//        * High 0.9ms -> AUTO mode
//   3) STOP mode is not used; AUTO should output neutral when it has no data.
//   4) Onboard RGB LED indicates mode:
//        * RED (PF1) -> MANUAL mode
//        * BLUE (PF2) -> AUTO mode

#include "control.h"
#include "manual.h"
#include "auto.h"
#include "pwm_output.h"
#include "rc_input.h"
#include "tm4c123gh6pm.h"

// LED pins on TM4C123GH6PM LaunchPad (Port F)
#define LED_RED    (1U << 1)   // PF1
#define LED_BLUE   (1U << 2)   // PF2
#define LED_GREEN  (1U << 3)   // PF3

static control_mode_t g_curr_mode = CONTROL_MODE_MANUAL;

// -----------------------------------------------------------
// Helper: Update LED to indicate current driving mode
//   RED (PF1)  -> MANUAL mode
//   BLUE (PF2) -> AUTO mode
// -----------------------------------------------------------
static void Control_UpdateLED(void)
{
    // Turn off all LEDs first, then turn on the appropriate one
    GPIO_PORTF_DATA_R &= ~(LED_RED | LED_BLUE | LED_GREEN);
    
    if (g_curr_mode == CONTROL_MODE_MANUAL) {
        // MANUAL mode: RED LED on
        GPIO_PORTF_DATA_R |= LED_RED;
    } else {
        // AUTO mode: BLUE LED on
        GPIO_PORTF_DATA_R |= LED_BLUE;
    }
}

// -----------------------------------------------------------
// Helper: Update mode based on RC mode switch (PB2)
//   Returns 1 if mode changed, 0 otherwise
// -----------------------------------------------------------
static uint8_t Control_UpdateMode(void)
{
    control_mode_t new_mode;
    
    // Check mode switch: 1 = MANUAL, 0 = AUTO
    if (RC_IsManualMode()) {
        new_mode = CONTROL_MODE_MANUAL;
    } else {
        new_mode = CONTROL_MODE_AUTO;
    }
    
    // Update mode if changed
    if (new_mode != g_curr_mode) {
        g_curr_mode = new_mode;
        Control_UpdateLED();  // Update LED when mode changes
        return 1U;
    }
    
    return 0U;
}

// -----------------------------------------------------------
// Public API
// -----------------------------------------------------------

void Control_Init(void)
{
    volatile uint32_t delay;
    
    // Initialize mode to MANUAL (fail-safe default)
    g_curr_mode = CONTROL_MODE_MANUAL;
    
    // Initialize GPIO Port F for LED control
    SYSCTL_RCGCGPIO_R |= (1U << 5);   // Enable Port F clock (bit 5)
    
    // Wait for Port F to be ready (small delay for clock stabilization)
    delay = SYSCTL_RCGCGPIO_R;
    delay = SYSCTL_RCGCGPIO_R;
    delay = SYSCTL_RCGCGPIO_R;
    (void)delay;
    
    // Disable analog mode first
    GPIO_PORTF_AMSEL_R &= ~(LED_RED | LED_BLUE | LED_GREEN);
    
    // Disable alternate functions
    GPIO_PORTF_AFSEL_R &= ~(LED_RED | LED_BLUE | LED_GREEN);
    
    // Clear PCTL for these pins (GPIO mode)
    GPIO_PORTF_PCTL_R &= ~(0x0000FFF0U);  // Clear PCTL for PF1, PF2, PF3
    
    // Configure PF1 (RED), PF2 (BLUE), PF3 (GREEN) as outputs
    GPIO_PORTF_DIR_R |= (LED_RED | LED_BLUE | LED_GREEN);
    
    // Enable digital function
    GPIO_PORTF_DEN_R |= (LED_RED | LED_BLUE | LED_GREEN);
    
    // Turn off all LEDs first
    GPIO_PORTF_DATA_R &= ~(LED_RED | LED_BLUE | LED_GREEN);
    
    // Set initial LED state (RED for MANUAL mode)
    Control_UpdateLED();
}

control_mode_t Control_GetMode(void)
{
    return g_curr_mode;
}

#define UART_DEBUG
void Control_RunFrame(void)
{
#	ifdef UART_DEBUG
	Auto_RunFrame();
#	else
    // 1) Update MANUAL/AUTO selection based on RC mode switch (PB2)
    Control_UpdateMode();

    // 2) Run control frame based on current mode
    switch (g_curr_mode)
    {
    case CONTROL_MODE_MANUAL:
        Manual_RunFrame();
        break;

    case CONTROL_MODE_AUTO:
        Auto_RunFrame();
        break;
    
    default:
        // Fail-safe: default to MANUAL
        Manual_RunFrame();
        break;
    }
    // NOTE:
    //  - STOP mode is not used here. If AUTO has no valid data,
    //    Auto_RunFrame() should output neutral (safe stop) itself.
#	endif
}
