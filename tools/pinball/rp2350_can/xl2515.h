/*
 * XL2515 SPI driver for the Waveshare RP2350-CAN.
 * Derived from Waveshare's RP2350-CAN-Demo C source (downloaded 2026-09-12).
 * The controller is register compatible with the MCP2515. This source is
 * included intentionally: no prebuilt vendor firmware or binary library is
 * required.
 */
#ifndef PINBALL_XL2515_H
#define PINBALL_XL2515_H

#include <stdbool.h>
#include <stdint.h>

bool xl2515_init_500k(void);
bool xl2515_receive(uint16_t *standard_id, uint8_t data[8], uint8_t *length);
bool xl2515_send(uint16_t standard_id, const uint8_t *data, uint8_t length);
uint8_t xl2515_error_flags(void);

#endif
