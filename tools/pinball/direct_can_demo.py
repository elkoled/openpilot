#!/usr/bin/env python3
"""Quick bench demo: WebRTC testJoystick directly to Panda CAN."""

import signal
import time

from opendbc.car.structs import CarParams
from openpilot.cereal import messaging
from panda import Panda

try:
  from openpilot.tools.pinball.protocol import COMMAND_ID, STATUS_ID, command, parse_status, pressed_axes
except ImportError:  # standalone /tmp deployment on an unchanged comma checkout
  from pinball_protocol import COMMAND_ID, STATUS_ID, command, parse_status, pressed_axes


def main() -> None:
  panda = Panda()
  panda.set_power_save(False)
  panda.set_can_speed_kbps(0, 500)
  panda.set_safety_mode(CarParams.SafetyModel.allOutput)
  joystick = messaging.SubMaster(["testJoystick"])
  running = True

  def stop(_signum, _frame):
    nonlocal running
    running = False

  signal.signal(signal.SIGINT, stop)
  signal.signal(signal.SIGTERM, stop)
  sequence = 0
  left = right = start = False
  last_ack_ns = 0
  last_heartbeat_ns = 0
  rx_count = 0
  print("PINBALL CAN READY: bus 0, 500 kbit/s, command 0x200, status 0x201", flush=True)

  try:
    while running:
      start_ns = time.monotonic_ns()
      joystick.update(0)
      if joystick.updated["testJoystick"]:
        left, right, start = pressed_axes(joystick["testJoystick"].axes)

      panda.can_send(COMMAND_ID, command(sequence, left, right, start), 0)
      if start_ns - last_heartbeat_ns >= 500_000_000:
        panda.send_heartbeat(False)
        last_heartbeat_ns = start_ns

      for address, data, source in panda.can_recv():
        if address == STATUS_ID and source == 0:
          status = parse_status(bytes(data))
          if status is not None:
            rx_count = status["rx_count"]
            last_ack_ns = time.monotonic_ns()

      if sequence % 100 == 0:
        ack_age_ms = (start_ns - last_ack_ns) / 1e6 if last_ack_ns else -1
        print(f"state={int(left)}{int(right)}{int(start)} seq={sequence} rp_rx={rx_count} ack_age_ms={ack_age_ms:.1f}", flush=True)
      sequence = (sequence + 1) & 0xFF
      delay = 0.01 - (time.monotonic_ns() - start_ns) / 1e9
      if delay > 0:
        time.sleep(delay)
  finally:
    for _ in range(3):
      panda.can_send(COMMAND_ID, command(sequence, False, False, False), 0)
      sequence = (sequence + 1) & 0xFF
    panda.set_safety_mode(CarParams.SafetyModel.noOutput)
    print("PINBALL CAN STOPPED: release sent, Panda no-output restored", flush=True)


if __name__ == "__main__":
  main()
