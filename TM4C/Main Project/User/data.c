// data.c
// Generate high level commands: speed + steering.

#include "data.h"

static data_cmd_t g_cmd;
static uint32_t   g_frame_cnt = 0U;

void Data_Init(void)
{
    g_frame_cnt  = 0U;
    g_cmd.speed  = 0;
    g_cmd.steer  = 0;
    g_cmd.valid  = 0;
}

// Simple pattern (for test):
//  0-99   : no command (valid = 0)
//  100-299: forward straight (speed > 0, steer = 0)
//  300-499: forward left      (speed > 0, steer < 0)
//  500-699: forward right     (speed > 0, steer > 0)
//  700-799: stop              (speed = 0, steer = 0)
void Data_RunFrame(void)
{
    g_frame_cnt++;
    uint32_t t = g_frame_cnt % 800U;

    if (t < 100U) {
        g_cmd.valid = 0U;
        g_cmd.speed = 0;
        g_cmd.steer = 0;
    }
    else if (t < 300U) {
        g_cmd.valid = 1U;
        g_cmd.speed = 700;   // forward
        g_cmd.steer = 0;     // straight
    }
    else if (t < 500U) {
        g_cmd.valid = 1U;
        g_cmd.speed = 600;   // forward
        g_cmd.steer = -600;  // left
    }
    else if (t < 700U) {
        g_cmd.valid = 1U;
        g_cmd.speed = -600;   // forward
        g_cmd.steer = 600;   // right
    }
    else {
        g_cmd.valid = 1U;
        g_cmd.speed = 0;     // stop
        g_cmd.steer = 0;
    }
}

uint8_t Data_GetCmd(data_cmd_t *cmd)
{
    if (!cmd) return 0U;

    cmd->speed = g_cmd.speed;
    cmd->steer = g_cmd.steer;
    cmd->valid = g_cmd.valid;

    return (g_cmd.valid ? 1U : 0U);
}
