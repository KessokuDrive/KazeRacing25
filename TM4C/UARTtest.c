// uart1_test_standalone.c
// Standalone UART1 test program for TM4C123
// Tests UART1 TX and RX on PB0 (RX) and PB1 (TX)
//
// Functionality:
// 1. Echoes back any received data (loopback test)
// 2. Sends periodic test messages every 1 second
// 3. Blinks LED to show activity
//
// Usage:
// - Flash this program to TM4C
// - Connect UART1: PB0 (RX), PB1 (TX), GND
// - Use serial terminal at 115200 baud
// - Type something, should echo back
// - Should see "UART1 TEST X" messages every second

#include <stdint.h>
#include <stdbool.h>

// TivaWare includes
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/uart.h"
#include "driverlib/pin_map.h"

// Simple delay
void delay_ms(uint32_t ms)
{
    // Approximate delay at 80MHz
    SysCtlDelay((SysCtlClockGet() / 3000) * ms);
}

// Send string over UART1
void uart1_puts(const char *str)
{
    while(*str) {
        UARTCharPut(UART1_BASE, *str++);
    }
}

// Send hex byte as ASCII
void uart1_put_hex(uint8_t byte)
{
    const char hex[] = "0123456789ABCDEF";
    UARTCharPut(UART1_BASE, hex[byte >> 4]);
    UARTCharPut(UART1_BASE, hex[byte & 0x0F]);
}

// Initialize LED on PF1 (Red LED)
void led_init(void)
{
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF)) {}
    
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3);
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3, 0);
}

// Toggle LED
void led_toggle(void)
{
    static uint8_t state = 0;
    state = !state;
    GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_1, state ? GPIO_PIN_1 : 0);
}

// Initialize UART1 on PB0 (RX) and PB1 (TX)
void uart1_init(void)
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
    
    // Enable UART
    UARTEnable(UART1_BASE);
}

int main(void)
{
    uint32_t counter = 0;
    uint32_t rx_count = 0;
    
    // Set system clock to 80MHz
    SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | 
                   SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);
    
    // Initialize LED
    led_init();
    
    // Initialize UART1
    uart1_init();
    
    // Startup message
    delay_ms(100);
    uart1_puts("\r\n");
    uart1_puts("========================================\r\n");
    uart1_puts("TM4C123 UART1 Test Program\r\n");
    uart1_puts("========================================\r\n");
    uart1_puts("UART1: PB0 (RX), PB1 (TX)\r\n");
    uart1_puts("Baud: 115200, 8-N-1\r\n");
    uart1_puts("\r\n");
    uart1_puts("Features:\r\n");
    uart1_puts("- Echo mode (type to see echo)\r\n");
    uart1_puts("- Periodic test messages\r\n");
    uart1_puts("- LED blinks on activity\r\n");
    uart1_puts("\r\n");
    uart1_puts("Ready!\r\n");
    uart1_puts("========================================\r\n");
    uart1_puts("\r\n");
    
    // Main loop
    while(1)
    {
        // Check for received data
        if(UARTCharsAvail(UART1_BASE))
        {
            // Read character
            int32_t ch = UARTCharGetNonBlocking(UART1_BASE);
            
            if(ch != -1)
            {
                rx_count++;
                
                // Echo back the character
                UARTCharPut(UART1_BASE, (uint8_t)ch);
                
                // Also send in hex format for debugging
                uart1_puts(" [0x");
                uart1_put_hex((uint8_t)ch);
                uart1_puts("] ");
                
                // Toggle LED to show activity
                led_toggle();
                
                // Special handling for Enter/newline
                if(ch == '\r' || ch == '\n') {
                    uart1_puts("\r\n");
                    uart1_puts("Received: ");
                    uart1_put_hex((rx_count >> 8) & 0xFF);
                    uart1_put_hex(rx_count & 0xFF);
                    uart1_puts(" chars total\r\n");
                }
            }
        }
        
        // Send periodic test message (every ~1 second)
        counter++;
        if(counter >= 40000)  // Adjust based on actual timing
        {
            counter = 0;
            
            // Send test message
            uart1_puts(">>> UART1 TEST ");
            
            // Send counter as hex
            static uint32_t msg_count = 0;
            msg_count++;
            uart1_put_hex((msg_count >> 8) & 0xFF);
            uart1_put_hex(msg_count & 0xFF);
            
            uart1_puts(" <<<\r\n");
            
            // Toggle LED
            led_toggle();
        }
        
        // Small delay
        SysCtlDelay(100);
    }
}
