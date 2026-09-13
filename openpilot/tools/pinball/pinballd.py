#!/usr/bin/env python3
"""Bridge WebRTC testJoystick messages to the dedicated pinball CAN contract."""

import time

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.pandad import can_list_to_can_capnp
from opendbc.car.can_definitions import CanData

from openpilot.tools.pinball.protocol import CAN_BUS, COMMAND_ID, command, pressed_axes

INPUT_TIMEOUT_NS = 150_000_000


def main() -> None:
  sm = messaging.SubMaster(["testJoystick"])
  pm = messaging.PubMaster(["sendcan"])
  rk = Ratekeeper(100, print_delay_threshold=None)
  sequence = 0
  last_input_ns = 0
  left = right = False

  while True:
    sm.update(0)
    now = time.monotonic_ns()
    if sm.updated["testJoystick"]:
      left, right = pressed_axes(sm["testJoystick"].axes)
      last_input_ns = now
    elif last_input_ns == 0 or now - last_input_ns > INPUT_TIMEOUT_NS:
      left = right = False

    # A 100 Hz heartbeat gives the RP controller an independent 200 ms watchdog.
    payload = command(sequence, left, right)
    pm.send("sendcan", can_list_to_can_capnp([CanData(COMMAND_ID, payload, CAN_BUS)], msgtype="sendcan"))

    sequence = (sequence + 1) & 0xFF
    rk.keep_time()


if __name__ == "__main__":
  main()
