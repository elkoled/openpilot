COMMAND_ID = 0x200
STATUS_ID = 0x201
WATCHDOG_MS = 200

LEFT_PRESSED = 1 << 0
RIGHT_PRESSED = 1 << 1
START_PRESSED = 1 << 2
STATE_MASK = LEFT_PRESSED | RIGHT_PRESSED | START_PRESSED
STATUS_WATCHDOG = 1 << 7

FAULT_BAD_FRAME = 1 << 0
FAULT_STALE_SEQUENCE = 1 << 1
FAULT_WATCHDOG = 1 << 2
FAULT_CAN = 1 << 3


def crc8(data):
  crc = 0xFF
  for value in data:
    crc ^= value
    for _ in range(8):
      crc = ((crc << 1) ^ (0x1D if crc & 0x80 else 0)) & 0xFF
  return crc ^ 0xFF


class Controller:
  def __init__(self):
    self.state = 0
    self.last_sequence = 0
    self.faults = 0
    self.rx_count = 0
    self.have_sequence = False
    self.watchdog_active = False
    self.last_command_ms = 0

  def _accept(self, state, now_ms):
    self.state = state
    self.last_command_ms = now_ms
    self.have_sequence = True
    self.watchdog_active = False
    self.rx_count = (self.rx_count + 1) & 0xFFFF
    self.faults &= ~(FAULT_BAD_FRAME | FAULT_STALE_SEQUENCE | FAULT_WATCHDOG)
    return True

  def apply(self, data, now_ms):
    # Deployed sender: one byte containing the live button state.
    if len(data) == 1:
      if data[0] & ~STATE_MASK:
        self.faults |= FAULT_BAD_FRAME
        return False
      return self._accept(data[0], now_ms)

    # Backward-compatible protected command frame.
    if (len(data) != 8 or data[0:3] != b"PB\x01" or
        data[4] & ~STATE_MASK or data[5] != ((~data[4]) & 0xFF) or
        data[6] != 0 or data[7] != crc8(data[:7])):
      self.faults |= FAULT_BAD_FRAME
      return False
    sequence = data[3]
    if self.have_sequence:
      advance = (sequence - self.last_sequence) & 0xFF
      if advance == 0 or advance > 127:
        self.faults |= FAULT_STALE_SEQUENCE
        return False
    self.last_sequence = sequence
    return self._accept(data[4], now_ms)

  def poll_watchdog(self, now_ms, ticks_diff=None):
    elapsed = (ticks_diff(now_ms, self.last_command_ms) if ticks_diff else
               ((now_ms - self.last_command_ms) & 0xFFFFFFFF))
    if self.have_sequence and elapsed >= WATCHDOG_MS:
      self.state = 0
      self.have_sequence = False
      self.watchdog_active = True
      self.faults |= FAULT_WATCHDOG
      return True
    return False

  def status(self):
    out = bytearray((0x50, 0x42, 1, self.last_sequence,
                     self.state | (STATUS_WATCHDOG if self.watchdog_active else 0),
                     self.rx_count & 0xFF, self.rx_count >> 8, 0))
    out[7] = crc8(out[:7])
    return out
