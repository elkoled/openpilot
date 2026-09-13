from openpilot.tools.pinball.protocol import command, crc8, parse_status, pressed_axes


def test_command_contract():
  payload = command(0xA5, True, False)
  assert payload[:6] == bytes((0x50, 0x42, 1, 0xA5, 1, 0xFE))
  assert len(payload) == 8
  assert crc8(payload[:7]) == payload[7]


def test_all_button_states():
  assert [command(i, i & 1, i & 2)[4] for i in range(4)] == [0, 1, 2, 3]


def test_axes_are_strict_and_missing_axes_release():
  assert pressed_axes([]) == (False, False)
  assert pressed_axes([0.5, 0.50001]) == (False, True)


def test_status_validation():
  raw = bytearray((0x50, 0x42, 1, 7, 0x82, 0x34, 0x12, 0))
  raw[7] = crc8(raw[:7])
  assert parse_status(bytes(raw)) == {
    "sequence": 7, "left_pressed": False, "right_pressed": True,
    "watchdog_released": True, "rx_count": 0x1234,
  }
  raw[4] ^= 1
  assert parse_status(bytes(raw)) is None
