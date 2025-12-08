// auto.c
// AUTO mode:
//  - read speed + steering from upper board (simulated by data.c)
//  - map to ESC / servo pulse width
//  - send PWM via pwm_output.c

#include "auto.h"
#include "data.h"
#include "pwm_output.h"

#define NEUTRAL_US     1500U
#define ESC_MIN_US     500U
#define ESC_MAX_US     1000U
#define SERVO_MIN_US   1100U   // smaller range for servo
#define SERVO_MAX_US   1900U

// speed, steer range from upper board: -1000..+1000
#define CMD_RANGE      1000

static uint16_t map_speed_to_pwm(int16_t speed)
{
    // Clamp to [-CMD_RANGE, CMD_RANGE]
    if (speed > CMD_RANGE)  speed = CMD_RANGE;
    if (speed < -CMD_RANGE) speed = -CMD_RANGE;

    // Map to ESC pulse:
    //  speed = 0      -> NEUTRAL
    //  speed = +1000  -> ESC_MAX_US
    //  speed = -1000  -> ESC_MIN_US
    int32_t delta = (int32_t)(ESC_MAX_US - ESC_MIN_US) * speed / (2 * CMD_RANGE);
    int32_t pwm   = (int32_t)NEUTRAL_US + delta;

    if (pwm < (int32_t)ESC_MIN_US) pwm = ESC_MIN_US;
    if (pwm > (int32_t)ESC_MAX_US) pwm = ESC_MAX_US;

    return (uint16_t)pwm;
}

static uint16_t map_steer_to_pwm(int16_t steer)
{
    // Clamp to [-CMD_RANGE, CMD_RANGE]
    if (steer > CMD_RANGE)  steer = CMD_RANGE;
    if (steer < -CMD_RANGE) steer = -CMD_RANGE;

    // Map to servo pulse:
    //  steer = 0     -> NEUTRAL
    //  steer = +1000 -> SERVO_MAX_US (right)
    //  steer = -1000 -> SERVO_MIN_US (left)
    int32_t delta = (int32_t)(SERVO_MAX_US - SERVO_MIN_US) * steer / (2 * CMD_RANGE);
    int32_t pwm   = (int32_t)NEUTRAL_US + delta;

    if (pwm < (int32_t)SERVO_MIN_US) pwm = SERVO_MIN_US;
    if (pwm > (int32_t)SERVO_MAX_US) pwm = SERVO_MAX_US;

    return (uint16_t)pwm;
}

void Auto_Init(void)
{
    // Init data simulator or real upper-board interface.
    Data_Init();
}

void Auto_RunFrame(void)
{
    data_cmd_t cmd;
    uint8_t has_cmd;

    // Update internal command pattern (or receive new packet).
    Data_RunFrame();

    // Read latest high-level command.
    has_cmd = Data_GetCmd(&cmd);

    if (has_cmd && cmd.valid) {
        uint16_t motor_us = map_speed_to_pwm(cmd.speed);
        uint16_t steer_us = map_steer_to_pwm(cmd.steer);

        PWM_Output_WriteFrame(motor_us, steer_us);
    } else {
        // No valid command -> neutral safe output.
        PWM_Output_WriteFrame(NEUTRAL_US, NEUTRAL_US);
    }
}
