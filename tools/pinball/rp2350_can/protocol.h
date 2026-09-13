#ifndef PINBALL_PROTOCOL_H
#define PINBALL_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PINBALL_COMMAND_ID 0x200u
#define PINBALL_STATUS_ID 0x201u
#define PINBALL_FRAME_LEN 8u
#define PINBALL_WATCHDOG_MS 200u
#define PINBALL_PROTOCOL_MAGIC_0 0x50u
#define PINBALL_PROTOCOL_MAGIC_1 0x42u
#define PINBALL_PROTOCOL_VERSION 1u
#define PINBALL_STATUS_WATCHDOG (1u << 7)

#define PINBALL_LEFT_PRESSED  (1u << 0)
#define PINBALL_RIGHT_PRESSED (1u << 1)
#define PINBALL_START_PRESSED (1u << 2)
#define PINBALL_STATE_MASK (PINBALL_LEFT_PRESSED | PINBALL_RIGHT_PRESSED | PINBALL_START_PRESSED)

enum pinball_fault {
  PINBALL_FAULT_NONE = 0,
  PINBALL_FAULT_BAD_FRAME = 1u << 0,
  PINBALL_FAULT_STALE_SEQUENCE = 1u << 1,
  PINBALL_FAULT_WATCHDOG = 1u << 2,
  PINBALL_FAULT_CAN = 1u << 3,
};

typedef struct {
  uint8_t state;
  uint8_t last_sequence;
  uint8_t faults;
  uint16_t rx_count;
  bool have_sequence;
  bool watchdog_active;
  uint32_t last_command_ms;
} pinball_controller_t;

uint8_t pinball_crc8(const uint8_t *data, size_t len);
void pinball_make_command(uint8_t sequence, uint8_t state, uint8_t out[PINBALL_FRAME_LEN]);
bool pinball_command_valid(const uint8_t frame[PINBALL_FRAME_LEN]);
bool pinball_apply_command(pinball_controller_t *controller,
                           const uint8_t frame[PINBALL_FRAME_LEN], uint32_t now_ms);
bool pinball_watchdog_poll(pinball_controller_t *controller, uint32_t now_ms);
void pinball_make_status(const pinball_controller_t *controller,
                         uint8_t out[PINBALL_FRAME_LEN]);

#endif
