// jetson_control.c
// UART1 communication with Jetson Nano for autonomous driving commands.
// Protocol: 6-byte packet [0xAA][speed_low][speed_high][steer_low][steer_high][checksum]
// Baud rate: 115200, 8N1

#include "jetson_control.h"
#include <stdint.h>
#include <stdbool.h>

// TivaWare includes
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_ints.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/uart.h"
#include "driverlib/pin_map.h"
#include "driverlib/interrupt.h"

// Protocol definitions
#define PACKET_HEADER     0xAA
#define PACKET_SIZE       6
#define ACK_HEADER        0x55
#define ACK_SIZE          6

// Status codes for acknowledgment
#define STATUS_OK         0x00
#define STATUS_BAD_CSUM   0x01
#define STATUS_TIMEOUT    0x02

// Packet state machine
typedef enum {
    STATE_WAIT_HEADER = 0,
    STATE_SPEED_LOW,
    STATE_SPEED_HIGH,
    STATE_STEER_LOW,
    STATE_STEER_HIGH,
    STATE_CHECKSUM
} rx_state_t;

// Receive buffer and state
static volatile rx_state_t g_rx_state = STATE_WAIT_HEADER;
static volatile uint8_t g_rx_buffer[PACKET_SIZE];
static volatile uint8_t g_rx_index = 0;

// Latest valid command
static volatile data_cmd_t g_jetson_cmd;
static volatile uint8_t g_cmd_updated = 0;
static volatile uint32_t g_last_rx_time = 0;

// Timeout tracking (in ms, approximate)
#define CMD_TIMEOUT_MS    500U
static volatile uint32_t g_tick_count = 0;

// Statistics
static volatile uint32_t g_packets_received = 0;
static volatile uint32_t g_packets_bad_checksum = 0;

//-----------------------------------------------------------------------------
// Send acknowledgment packet to Jetson
//-----------------------------------------------------------------------------
static void send_ack(int16_t speed, int16_t steer, uint8_t status)
{
    uint8_t ack_packet[ACK_SIZE];
    
    // Build acknowledgment packet
    ack_packet[0] = ACK_HEADER;
    ack_packet[1] = (uint8_t)(speed & 0xFF);        // Speed low byte
    ack_packet[2] = (uint8_t)((speed >> 8) & 0xFF); // Speed high byte
    ack_packet[3] = (uint8_t)(steer & 0xFF);        // Steer low byte
    ack_packet[4] = (uint8_t)((steer >> 8) & 0xFF); // Steer high byte
    ack_packet[5] = status;                          // Status byte
    
    // Send each byte
    for(uint8_t i = 0; i < ACK_SIZE; i++) {
        UARTCharPut(UART1_BASE, ack_packet[i]);
    }
}

//-----------------------------------------------------------------------------
// UART1 Interrupt Handler
//-----------------------------------------------------------------------------
void UART1IntHandler(void)
{
    uint32_t ui32Status;
    int32_t rx_char;
    
    // Get and clear the interrupt status
    ui32Status = UARTIntStatus(UART1_BASE, true);
    UARTIntClear(UART1_BASE, ui32Status);
    
    // Process all available characters
    while(UARTCharsAvail(UART1_BASE))
    {
        rx_char = UARTCharGetNonBlocking(UART1_BASE);
        
        if(rx_char == -1) {
            break;  // No more characters
        }
        
        uint8_t byte = (uint8_t)rx_char;
        
        switch(g_rx_state)
        {
            case STATE_WAIT_HEADER:
                if(byte == PACKET_HEADER) {
                    g_rx_buffer[0] = byte;
                    g_rx_index = 1;
                    g_rx_state = STATE_SPEED_LOW;
                }
                break;
                
            case STATE_SPEED_LOW:
                g_rx_buffer[1] = byte;
                g_rx_index = 2;
                g_rx_state = STATE_SPEED_HIGH;
                break;
                
            case STATE_SPEED_HIGH:
                g_rx_buffer[2] = byte;
                g_rx_index = 3;
                g_rx_state = STATE_STEER_LOW;
                break;
                
            case STATE_STEER_LOW:
                g_rx_buffer[3] = byte;
                g_rx_index = 4;
                g_rx_state = STATE_STEER_HIGH;
                break;
                
            case STATE_STEER_HIGH:
                g_rx_buffer[4] = byte;
                g_rx_index = 5;
                g_rx_state = STATE_CHECKSUM;
                break;
                
            case STATE_CHECKSUM:
            {
                // Add braces to create proper scope for local variables
                g_rx_buffer[5] = byte;
                
                // Validate checksum (XOR of bytes 1-4)
                uint8_t calc_checksum = g_rx_buffer[1] ^ g_rx_buffer[2] ^ 
                                       g_rx_buffer[3] ^ g_rx_buffer[4];
                
                // Decode speed and steer
                int16_t speed = (int16_t)((g_rx_buffer[2] << 8) | g_rx_buffer[1]);
                int16_t steer = (int16_t)((g_rx_buffer[4] << 8) | g_rx_buffer[3]);
                
                if(calc_checksum == byte) {
                    // Valid packet
                    g_packets_received++;
                    
                    // Clamp values to valid range
                    if(speed > 1000) speed = 1000;
                    if(speed < -1000) speed = -1000;
                    if(steer > 1000) steer = 1000;
                    if(steer < -1000) steer = -1000;
                    
                    // Update command
                    g_jetson_cmd.speed = speed;
                    g_jetson_cmd.steer = steer;
                    g_jetson_cmd.valid = 1;
                    g_cmd_updated = 1;
                    g_last_rx_time = g_tick_count;
                    
                    // Send acknowledgment
                    send_ack(speed, steer, STATUS_OK);
                } else {
                    // Bad checksum
                    g_packets_bad_checksum++;
                    send_ack(speed, steer, STATUS_BAD_CSUM);
                }
                
                // Reset state machine
                g_rx_state = STATE_WAIT_HEADER;
                g_rx_index = 0;
                break;
            }
                
            default:
                // Invalid state - reset
                g_rx_state = STATE_WAIT_HEADER;
                g_rx_index = 0;
                break;
        }
    }
}

//-----------------------------------------------------------------------------
// Initialize UART1 for Jetson communication
//-----------------------------------------------------------------------------
void UART_Vision_Init(void)
{
    // Enable peripherals
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART1);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    
    // Wait for peripherals to be ready
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_UART1)) {}
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOB)) {}
    
    // Configure GPIO pins for UART1
    // PB0 = U1RX, PB1 = U1TX
    GPIOPinConfigure(GPIO_PB0_U1RX);
    GPIOPinConfigure(GPIO_PB1_U1TX);
    GPIOPinTypeUART(GPIO_PORTB_BASE, GPIO_PIN_0 | GPIO_PIN_1);
    
    // Configure UART: 115200 baud, 8-N-1
    UARTConfigSetExpClk(UART1_BASE, SysCtlClockGet(), 115200,
                        (UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE |
                         UART_CONFIG_PAR_NONE));
    
    // Enable UART1 RX interrupt
    IntEnable(INT_UART1);
	UARTEnable(UART1_BASE);
    UARTIntEnable(UART1_BASE, UART_INT_RX | UART_INT_RT);
    
    // Initialize command structure
    g_jetson_cmd.speed = 0;
    g_jetson_cmd.steer = 0;
    g_jetson_cmd.valid = 0;
    g_cmd_updated = 0;
    g_last_rx_time = 0;
    g_tick_count = 0;
    
    // Reset state machine
    g_rx_state = STATE_WAIT_HEADER;
    g_rx_index = 0;
    
    // Enable UART
    UARTEnable(UART1_BASE);
}

//-----------------------------------------------------------------------------
// Process receive timeout and update command validity
//-----------------------------------------------------------------------------
void Jetson_ProcessRx(void)
{
    // Increment tick counter (called every ~20ms, so 1 tick ~ 20ms)
    g_tick_count++;
    
    // Check for timeout (no data received for CMD_TIMEOUT_MS)
    uint32_t elapsed = (g_tick_count - g_last_rx_time) * 20;  // Convert to ms
    
    if(elapsed > CMD_TIMEOUT_MS && g_jetson_cmd.valid) {
        // Timeout - invalidate command for safety
        g_jetson_cmd.valid = 0;
        g_jetson_cmd.speed = 0;
        g_jetson_cmd.steer = 0;
    }
}

//-----------------------------------------------------------------------------
// Get latest command from Jetson
//-----------------------------------------------------------------------------
uint8_t Jetson_GetCmd(data_cmd_t *cmd)
{
    if(!cmd) {
        return 0;
    }
    
    // Disable interrupts briefly to read atomic data
    IntMasterDisable();
    
    cmd->speed = g_jetson_cmd.speed;
    cmd->steer = g_jetson_cmd.steer;
    cmd->valid = g_jetson_cmd.valid;
    
    IntMasterEnable();
    
    return cmd->valid;
}
