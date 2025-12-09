// jetson_control.c
// UART0 communication with Jetson Nano
// Protocol: "T:####,S:####\n" (throttle and steer values)

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include "tm4c123gh6pm.h"
#include "jetson_control.h"
#include "data.h"

// UART0 uses PA0 (RX) and PA1 (TX)
#define UART0_BAUDRATE  115200
#define SYS_CLOCK_HZ    50000000UL

// Command timeout: if no valid command received in this many ms, invalidate
#define CMD_TIMEOUT_MS  500

// Receive buffer for incoming UART data
#define RX_BUFFER_SIZE  64
static char rx_buffer[RX_BUFFER_SIZE];
static volatile uint8_t rx_index = 0;

// Latest parsed command
static data_cmd_t latest_cmd = {0, 0, 0};  // speed=0, steer=0, valid=0
static volatile uint32_t last_cmd_time = 0;  // simple tick counter

// Simple millisecond counter (incremented externally if needed, or use SysTick)
static volatile uint32_t system_tick_ms = 0;

// Forward declarations
static void UART0_SendChar(char c);
void UART0_SendString(const char *str);
static bool UART0_DataAvailable(void);
static char UART0_ReadChar(void);
static void ParseCommand(const char *buffer);

//*****************************************************************************
// Initialize UART0 for communication with Jetson Nano
// UART0: PA0=RX, PA1=TX, 115200 baud, 8N1
//*****************************************************************************
void UART_Vision_Init(void)
{
    uint32_t brd_int, brd_frac;
    
    // 1. Enable clock to UART0 and GPIO Port A
    SYSCTL_RCGCUART_R |= 0x01;      // UART0
    SYSCTL_RCGCGPIO_R |= 0x01;      // Port A
    
    // Small delay for peripheral to be ready
    __asm("NOP"); __asm("NOP"); __asm("NOP");
    
    // 2. Disable UART0 while configuring
    UART0_CTL_R &= ~0x01;           // Clear UARTEN bit
    
    // Enable UART0 receive interrupt
    UART0_IM_R |= 0x10;             // Enable RX interrupt
    NVIC_EN0_R |= 0x20;             // Enable UART0 interrupt in NVIC
    NVIC_PRI1_R = (NVIC_PRI1_R & 0xFFFF00FF) | 0x00004000; // Priority 2
    
    // 3. Calculate baud rate divisor
    // BRD = BRDI + BRDF = UARTSysClk / (16 * Baud Rate)
    // For 50 MHz clock, 115200 baud:
    // BRD = 50000000 / (16 * 115200) = 27.1267
    // BRDI = 27, BRDF = int(0.1267 * 64 + 0.5) = 8
    brd_int = SYS_CLOCK_HZ / (16 * UART0_BAUDRATE);
    brd_frac = (uint32_t)(((float)(SYS_CLOCK_HZ % (16 * UART0_BAUDRATE)) / 
                          (16.0f * UART0_BAUDRATE)) * 64.0f + 0.5f);
    
    UART0_IBRD_R = brd_int;
    UART0_FBRD_R = brd_frac;
    
    // 4. Configure Line Control: 8 bits, no parity, 1 stop bit, FIFOs enabled
    UART0_LCRH_R = 0x60;            // 8-bit (WLEN=0x3), FIFO enable (FEN=1)
    
    // 5. Use system clock
    UART0_CC_R = 0x0;               // Use system clock
    
    // 6. Enable UART0 (TX, RX, and UART)
    UART0_CTL_R = 0x301;            // UARTEN=1, TXE=1, RXE=1
    
    // Enable interrupts globally
    __enable_irq();
    
    // 7. Configure GPIO PA0, PA1 for UART
    GPIO_PORTA_AFSEL_R |= 0x03;     // Enable alt function on PA0, PA1
    GPIO_PORTA_PCTL_R = (GPIO_PORTA_PCTL_R & 0xFFFFFF00) | 0x00000011; // UART
    GPIO_PORTA_DEN_R |= 0x03;       // Digital enable PA0, PA1
    GPIO_PORTA_AMSEL_R &= ~0x03;    // Disable analog on PA0, PA1
    
    // Initialize state
    rx_index = 0;
    latest_cmd.speed = 0;
    latest_cmd.steer = 0;
    latest_cmd.valid = 0;
    last_cmd_time = 0;
    system_tick_ms = 0;
    
    // Add a small delay to ensure UART is fully ready
    for (volatile uint32_t i = 0; i < 10000; i++);
    
    // Send init complete message
    UART0_SendString("TM4C UART0 INIT COMPLETE\r\n");
}

//*****************************************************************************
// Send a single character via UART0
//*****************************************************************************
static void UART0_SendChar(char c)
{
    // Wait until TX FIFO is not full
    while (UART0_FR_R & 0x20);      // Wait while TXFF=1
    UART0_DR_R = c;
}

//*****************************************************************************
// Send a string via UART0
//*****************************************************************************
void UART0_SendString(const char *str)
{
    while (*str) {
        UART0_SendChar(*str++);
    }
}

//*****************************************************************************
// Check if data is available in UART0 RX FIFO
//*****************************************************************************
static bool UART0_DataAvailable(void)
{
    return !(UART0_FR_R & 0x10);    // Return true if RXFE=0 (not empty)
}

//*****************************************************************************
// Read a character from UART0 (non-blocking, assumes data is available)
//*****************************************************************************
static char UART0_ReadChar(void)
{
    // Read directly from data register (caller should check availability first)
    return (char)(UART0_DR_R & 0xFF);
}

//*****************************************************************************
// Parse command string in format "T:####,S:####\n"
// Example: "T:500,S:-200\n"
//*****************************************************************************
static void ParseCommand(const char *buffer)
{
    int16_t throttle = 0;
    int16_t steer = 0;
    
    // Try to parse the command using sscanf
    if (sscanf(buffer, "T:%hd,S:%hd", &throttle, &steer) == 2) {
        // Valid parse - clamp values to expected range
        if (throttle > 1000) throttle = 1000;
        if (throttle < -1000) throttle = -1000;
        if (steer > 1000) steer = 1000;
        if (steer < -1000) steer = -1000;
        
        // Update latest command
        latest_cmd.speed = throttle;
        latest_cmd.steer = steer;
        latest_cmd.valid = 1;
        last_cmd_time = system_tick_ms;
        
        // Echo the received command back to Jetson with newline
        UART0_SendString("ACK:");
        UART0_SendString(buffer);
        UART0_SendString("\r\n");
    }
}

//*****************************************************************************
// UART0 interrupt handler
//*****************************************************************************
void UART0_Handler(void)
{
    // Clear interrupt
    UART0_ICR_R = 0x10;
    
    // Process received character
    char c = UART0_ReadChar();
    
    // Check for line ending (newline indicates complete command)
    if (c == '\n' || c == '\r') {
        if (rx_index > 0) {
            // Null-terminate and mark for parsing
            rx_buffer[rx_index] = '\0';
            rx_index = 0;  // Reset for next command
        }
    }
    // Check for buffer overflow
    else if (rx_index >= RX_BUFFER_SIZE - 1) {
        // Buffer full - discard and reset
        rx_index = 0;
    }
    // Normal character - add to buffer
    else {
        rx_buffer[rx_index++] = c;
    }
}

// Process received commands (call this periodically, e.g., every 20ms)
//*****************************************************************************
void Jetson_ProcessRx(void)
{
    // Increment tick counter (simple approach - 20ms per call)
    system_tick_ms += 20;
    
    // Check for command timeout
    if (latest_cmd.valid && 
        (system_tick_ms - last_cmd_time) > CMD_TIMEOUT_MS) {
        // Timeout - invalidate command for safety
        latest_cmd.valid = 0;
        latest_cmd.speed = 0;
        latest_cmd.steer = 0;
    }
    
    // Parse any complete commands in the buffer
    if (rx_buffer[0] != '\0') {
        ParseCommand(rx_buffer);
        // Clear buffer after parsing
        rx_buffer[0] = '\0';
    }
}

//*****************************************************************************
// Get the latest valid command from Jetson
// Returns 1 if valid command available, 0 otherwise
//*****************************************************************************
uint8_t Jetson_GetCmd(data_cmd_t *cmd)
{
    if (cmd == 0) {
        return 0;
    }
    
    // Copy latest command to caller's structure
    cmd->speed = latest_cmd.speed;
    cmd->steer = latest_cmd.steer;
    cmd->valid = latest_cmd.valid;
    
    return latest_cmd.valid;
}
