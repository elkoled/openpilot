"""Contract eval: malformed or stale states can never decode as a valid acknowledgement."""

from openpilot.tools.pinball.protocol import command, parse_status


def test_every_single_bit_error_is_rejected_as_status():
  original = command(9, True, True)
  for bit in range(64):
    damaged = bytearray(original)
    damaged[bit // 8] ^= 1 << (bit % 8)
    assert parse_status(bytes(damaged)) is None
