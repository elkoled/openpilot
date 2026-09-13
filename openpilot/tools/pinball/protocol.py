"""Wire contract shared by pinballd and its deterministic tests."""

COMMAND_ID = 0x200
STATUS_ID = 0x201
CAN_BUS = 0
MAGIC = (0x50, 0x42)  # "PB"
VERSION = 1


def pressed_axes(axes) -> tuple[bool, bool]:
  return (len(axes) > 0 and axes[0] > 0.5, len(axes) > 1 and axes[1] > 0.5)


def crc8(data: bytes) -> int:
  crc = 0xFF
  for value in data:
    crc ^= value
    for _ in range(8):
      crc = ((crc << 1) ^ 0x1D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
  return crc ^ 0xFF


def command(sequence: int, left: bool, right: bool) -> bytes:
  state = int(bool(left)) | (int(bool(right)) << 1)
  payload = bytes((*MAGIC, VERSION, sequence & 0xFF, state, state ^ 0xFF, 0))
  return payload + bytes((crc8(payload),))


def parse_status(payload: bytes) -> dict[str, int | bool] | None:
  if len(payload) != 8 or tuple(payload[:2]) != MAGIC or payload[2] != VERSION or crc8(payload[:7]) != payload[7]:
    return None
  state = payload[4]
  return {
    "sequence": payload[3],
    "left_pressed": bool(state & 1),
    "right_pressed": bool(state & 2),
    "watchdog_released": bool(state & 0x80),
    "rx_count": payload[5] | (payload[6] << 8),
  }
