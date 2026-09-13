#include "protocol.h"

uint8_t pinball_crc8(const uint8_t *data, size_t len) {
  uint8_t crc = 0xffu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (unsigned bit = 0; bit < 8; ++bit) {
      crc = (uint8_t)((uint8_t)(crc << 1) ^ ((crc & 0x80u) ? 0x1du : 0u));
    }
  }
  return (uint8_t)(crc ^ 0xffu);
}

void pinball_make_command(uint8_t sequence, uint8_t state, uint8_t out[8]) {
  state &= PINBALL_STATE_MASK;
  out[0] = PINBALL_PROTOCOL_MAGIC_0;
  out[1] = PINBALL_PROTOCOL_MAGIC_1;
  out[2] = PINBALL_PROTOCOL_VERSION;
  out[3] = sequence;
  out[4] = state;
  out[5] = (uint8_t)~state;
  out[6] = 0u;
  out[7] = pinball_crc8(out, 7);
}

bool pinball_command_valid(const uint8_t frame[8]) {
  return frame[0] == PINBALL_PROTOCOL_MAGIC_0 &&
         frame[1] == PINBALL_PROTOCOL_MAGIC_1 &&
         frame[2] == PINBALL_PROTOCOL_VERSION &&
         (frame[4] & (uint8_t)~PINBALL_STATE_MASK) == 0u &&
         frame[5] == (uint8_t)~frame[4] && frame[6] == 0u &&
         frame[7] == pinball_crc8(frame, 7);
}

bool pinball_apply_command(pinball_controller_t *controller,
                           const uint8_t frame[8], uint32_t now_ms) {
  if (!pinball_command_valid(frame)) {
    controller->faults |= PINBALL_FAULT_BAD_FRAME;
    return false;
  }

  if (controller->have_sequence) {
    const uint8_t advance = (uint8_t)(frame[3] - controller->last_sequence);
    if (advance == 0u || advance > 127u) {
      controller->faults |= PINBALL_FAULT_STALE_SEQUENCE;
      return false;
    }
  }

  controller->state = frame[4];
  controller->last_sequence = frame[3];
  controller->last_command_ms = now_ms;
  controller->have_sequence = true;
  controller->watchdog_active = false;
  controller->rx_count++;
  controller->faults &= (uint8_t)~(PINBALL_FAULT_BAD_FRAME |
                                  PINBALL_FAULT_STALE_SEQUENCE |
                                  PINBALL_FAULT_WATCHDOG);
  return true;
}

bool pinball_watchdog_poll(pinball_controller_t *controller, uint32_t now_ms) {
  if (controller->have_sequence &&
      (uint32_t)(now_ms - controller->last_command_ms) >= PINBALL_WATCHDOG_MS) {
    controller->state = 0u;
    controller->have_sequence = false;
    controller->watchdog_active = true;
    controller->faults |= PINBALL_FAULT_WATCHDOG;
    return true;
  }
  return false;
}

void pinball_make_status(const pinball_controller_t *controller, uint8_t out[8]) {
  out[0] = PINBALL_PROTOCOL_MAGIC_0;
  out[1] = PINBALL_PROTOCOL_MAGIC_1;
  out[2] = PINBALL_PROTOCOL_VERSION;
  out[3] = controller->last_sequence;
  out[4] = controller->state |
           (controller->watchdog_active ? PINBALL_STATUS_WATCHDOG : 0u);
  out[5] = (uint8_t)controller->rx_count;
  out[6] = (uint8_t)(controller->rx_count >> 8);
  out[7] = pinball_crc8(out, 7);
}
