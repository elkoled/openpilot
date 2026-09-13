#!/usr/bin/env python3
"""Bridge WebRTC testJoystick messages to the dedicated pinball CAN contract."""

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.pandad import can_list_to_can_capnp
from opendbc.car.can_definitions import CanData

try:
  from openpilot.tools.pinball.protocol import CAN_BUS, COMMAND_ID, command, pressed_axes
except ImportError:  # Standalone /tmp deployment for the hardware demo.
  from protocol import CAN_BUS, COMMAND_ID, command, pressed_axes

def main() -> None:
  sm = messaging.SubMaster(["testJoystick"])
  pm = messaging.PubMaster(["sendcan"])
  rk = Ratekeeper(100, print_delay_threshold=None)
  sequence = 0
  left = right = start = False

  while True:
    sm.update(0)
    if sm.updated["testJoystick"]:
      left, right, start = pressed_axes(sm["testJoystick"].axes)

    # A 100 Hz heartbeat gives the RP controller an independent 200 ms watchdog.
    payload = command(sequence, left, right, start)
    pm.send("sendcan", can_list_to_can_capnp([CanData(COMMAND_ID, payload, CAN_BUS)], msgtype="sendcan"))

    sequence = (sequence + 1) & 0xFF
    rk.keep_time()


if __name__ == "__main__":
  main()
