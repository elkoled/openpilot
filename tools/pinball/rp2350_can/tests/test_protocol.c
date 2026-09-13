#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "protocol.h"

static void test_known_vector(void) {
  uint8_t frame[8];
  pinball_make_command(0x2a, PINBALL_LEFT_PRESSED, frame);
  const uint8_t expected[8] = {0x50, 0x42, 0x01, 0x2a, 0x01, 0xfe, 0x00, 0xa8};
  assert(memcmp(frame, expected, sizeof(frame)) == 0);
  assert(pinball_command_valid(frame));
}

static void test_every_single_bit_corruption_is_rejected(void) {
  uint8_t good[8];
  pinball_make_command(7, PINBALL_LEFT_PRESSED | PINBALL_RIGHT_PRESSED, good);
  for (size_t byte = 0; byte < 8; ++byte) {
    for (unsigned bit = 0; bit < 8; ++bit) {
      uint8_t bad[8];
      memcpy(bad, good, 8);
      bad[byte] ^= (uint8_t)(1u << bit);
      assert(!pinball_command_valid(bad));
    }
  }
}

static void test_sequence_and_wrap(void) {
  pinball_controller_t c = {0};
  uint8_t frame[8];
  pinball_make_command(254, PINBALL_LEFT_PRESSED, frame);
  assert(pinball_apply_command(&c, frame, 10));
  pinball_make_command(255, PINBALL_RIGHT_PRESSED, frame);
  assert(pinball_apply_command(&c, frame, 11));
  pinball_make_command(0, 0, frame);
  assert(pinball_apply_command(&c, frame, 12));
  assert(!pinball_apply_command(&c, frame, 13));
  assert(c.faults & PINBALL_FAULT_STALE_SEQUENCE);
  assert(c.last_command_ms == 12);

  pinball_make_command(255, PINBALL_LEFT_PRESSED, frame);
  assert(!pinball_apply_command(&c, frame, 14));
  assert(c.state == 0u && c.last_sequence == 0u);
}

static void test_watchdog_and_millis_wrap(void) {
  pinball_controller_t c = {0};
  uint8_t frame[8];
  pinball_make_command(1, PINBALL_LEFT_PRESSED, frame);
  assert(pinball_apply_command(&c, frame, UINT32_MAX - 99u));
  assert(!pinball_watchdog_poll(&c, 99u));
  assert(pinball_watchdog_poll(&c, 100u));
  assert(c.state == 0u && !c.have_sequence);
  assert(c.faults & PINBALL_FAULT_WATCHDOG);
  assert(c.watchdog_active);
  assert(!pinball_watchdog_poll(&c, 101u));

  /* Watchdog clears sequence history so a restarted sender can resync. */
  pinball_make_command(0, PINBALL_RIGHT_PRESSED, frame);
  assert(pinball_apply_command(&c, frame, 102u));
  assert(!c.watchdog_active && c.state == PINBALL_RIGHT_PRESSED);
}

static void test_bad_frame_never_changes_output_or_refreshes_watchdog(void) {
  pinball_controller_t c = {0};
  uint8_t frame[8];
  pinball_make_command(3, PINBALL_LEFT_PRESSED, frame);
  assert(pinball_apply_command(&c, frame, 50));
  frame[1] ^= 1u;
  assert(!pinball_apply_command(&c, frame, 100));
  assert(c.state == PINBALL_LEFT_PRESSED && c.last_command_ms == 50);
  assert(pinball_watchdog_poll(&c, 250));
}

static void test_all_button_states_and_status(void) {
  for (uint8_t state = 0; state <= PINBALL_STATE_MASK; ++state) {
    pinball_controller_t c = {0};
    uint8_t frame[8];
    pinball_make_command((uint8_t)(40u + state), state, frame);
    assert(pinball_apply_command(&c, frame, 1));
    assert(c.state == state);
    assert(c.rx_count == 1u);
  }

  pinball_controller_t c = {.state = PINBALL_RIGHT_PRESSED,
                            .last_sequence = 9,
                            .rx_count = 0x1234,
                            .watchdog_active = true};
  uint8_t status[8];
  pinball_make_status(&c, status);
  assert(status[0] == 0x50 && status[1] == 0x42 && status[2] == 1);
  assert(status[3] == 9);
  assert(status[4] == (PINBALL_RIGHT_PRESSED | PINBALL_STATUS_WATCHDOG));
  assert(status[5] == 0x34 && status[6] == 0x12);
  assert(status[7] == pinball_crc8(status, 7));
}

int main(void) {
  test_known_vector();
  test_every_single_bit_corruption_is_rejected();
  test_sequence_and_wrap();
  test_watchdog_and_millis_wrap();
  test_bad_frame_never_changes_output_or_refreshes_watchdog();
  test_all_button_states_and_status();
  puts("protocol tests: PASS");
  return 0;
}
