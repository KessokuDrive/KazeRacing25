// auto.c
// AUTO mode:
//  - read speed + steering from upper board (simulated by data.c)
//  - map to ESC / servo pulse width
//  - send PWM via pwm_output.c

#include "auto.h"
#include "data.h"
#include "pwm_output.h"
#include "jetson_control.h"
#include <stdio.h>

#define NEUTRAL_US     1500U
#define ESC_MIN_US     1000U
#define ESC_MAX_US     2000U
#define SERVO_MIN_US   1100U   // smaller range for servo
#define SERVO_MAX_US   1900U

// speed, steer range from upper board: -1000..+1000
#define CMD_RANGE      1000

//Value of Motor and Servo in -1000+1000 scale
int16_t motor_val = 0;
int16_t steer_val = 0;

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
    motor_val = 0;
    steer_val = 0;
    UART_Vision_Init();
}

void Auto_RunFrame(void)
{
    data_cmd_t cmd;
    uint8_t has_cmd;

    // Process UART reception and check for timeout
    Jetson_ProcessRx();

    // Read latest high-level command from Jetson
    has_cmd = Jetson_GetCmd(&cmd);
    if (has_cmd && cmd.valid) {
        motor_val = cmd.speed;
        steer_val = cmd.steer;
    } else {
        // No valid command -> Keep privious
    }
		// Write the speed into pwm
		uint16_t motor_us = map_speed_to_pwm(motor_val);
		uint16_t steer_us = map_steer_to_pwm(steer_val);
		PWM_Output_WriteFrame(motor_us, steer_us);
		
}
