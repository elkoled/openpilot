from machine import PWM, Pin, SPI
from time import ticks_diff, ticks_ms

from protocol import COMMAND_ID, FAULT_BAD_FRAME, FAULT_CAN, STATUS_ID, Controller
from xl2515 import XL2515


LEFT_RELEASE_US = 1570
LEFT_PRESS_US = 1770
RIGHT_RELEASE_US = 1420
RIGHT_PRESS_US = 1220
START_RELEASE_US = 1420
START_PRESS_US = 1220


def servo(pin):
  output = PWM(Pin(pin))
  output.freq(333)
  return output


left = servo(14)
right = servo(15)
start = servo(4)


def apply_outputs(state):
  left.duty_ns((LEFT_PRESS_US if state & 1 else LEFT_RELEASE_US) * 1000)
  right.duty_ns((RIGHT_PRESS_US if state & 2 else RIGHT_RELEASE_US) * 1000)
  start.duty_ns((START_PRESS_US if state & 4 else START_RELEASE_US) * 1000)


controller = Controller()
apply_outputs(0)
spi = SPI(1, baudrate=10_000_000, polarity=0, phase=0,
          sck=Pin(10), mosi=Pin(11), miso=Pin(12))
can = XL2515(spi)
if not can.init_500k():
  controller.faults |= FAULT_CAN
  raise RuntimeError("CAN_INIT_FAILED faults=0x%02x" % controller.faults)

print("READY can=500000 command=0x200 status=0x201 state=0")
last_status_ms = ticks_ms()
last_log_ms = last_status_ms

while True:
  now_ms = ticks_ms()
  frame = can.receive()
  if frame is not None:
    can_id, data = frame
    accepted = can_id == COMMAND_ID and controller.apply(data, now_ms)
    if accepted:
      apply_outputs(controller.state)
    else:
      controller.faults |= FAULT_BAD_FRAME
    print("RX id=0x%03x dlc=%d data=%s accepted=%d state=%d faults=0x%02x" %
          (can_id, len(data), bytes(data).hex(), accepted, controller.state, controller.faults))
    if not can.send(STATUS_ID, controller.status()):
      controller.faults |= FAULT_CAN
    last_status_ms = now_ms

  if controller.poll_watchdog(now_ms, ticks_diff):
    apply_outputs(0)
    print("WATCHDOG_RELEASE faults=0x%02x" % controller.faults)
    can.send(STATUS_ID, controller.status())
    last_status_ms = now_ms

  if ticks_diff(now_ms, last_status_ms) >= 100:
    if can.error_flags():
      controller.faults |= FAULT_CAN
    can.send(STATUS_ID, controller.status())
    last_status_ms = now_ms
  if ticks_diff(now_ms, last_log_ms) >= 1000:
    print("ALIVE state=%d rx=%d faults=0x%02x" %
          (controller.state, controller.rx_count, controller.faults))
    last_log_ms = now_ms
