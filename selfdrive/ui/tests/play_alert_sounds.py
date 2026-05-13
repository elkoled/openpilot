#!/usr/bin/env python3
# Push to the comma device, run with openpilot's PYTHONPATH so msgq.ipc_pyx is
# found, with selfdrived stopped (so it doesn't overwrite our alertSound) and
# soundd alive (so it actually plays). This drives the REAL deployed soundd —
# not a re-implementation — so it can validate the get_sound_data fix on device.
#
#   scp selfdrive/ui/tests/play_alert_sounds.py comma@<device>:/data/openpilot/
#   ssh comma@<device>
#   tmux kill-session -t comma                  # stop the openpilot stack
#   cd /data/openpilot
#   ./selfdrive/ui/soundd.py &                  # run soundd alone
#   export PYTHONPATH=/data/openpilot:$PYTHONPATH
#   python3 selfdrive/ui/tests/play_alert_sounds.py laneChange         # loop one sound until Ctrl-C
#   python3 selfdrive/ui/tests/play_alert_sounds.py promptDistracted   # known-looping sound — best for wrap test
#   python3 selfdrive/ui/tests/play_alert_sounds.py                    # cycle every sound
#   python3 selfdrive/ui/tests/play_alert_sounds.py --seconds 30       # cycle, 30s each

import sys
import time

from cereal import messaging, car

AA = car.CarControl.HUDControl.AudibleAlert

ALL = ["engage", "disengage", "refuse", "prompt", "promptRepeat",
       "promptDistracted", "warningSoft", "warningImmediate", "laneChange"]


def keep_playing(name: str, pm: messaging.PubMaster, seconds: float | None) -> None:
  """Publish alertSound=<name> repeatedly so soundd keeps it active.

  For sounds with play_count=1 (engage/disengage/refuse/prompt/laneChange),
  soundd plays them ONCE per alertSound transition. To make them repeat, we
  briefly publish alertSound=none between repeats, forcing a fresh trigger.
  """
  if not hasattr(AA, name):
    print(f"  !! {name} not in cereal schema — stale build on device?")
    return

  one_shot = name in ("engage", "disengage", "refuse", "prompt", "laneChange")
  forever = seconds is None
  print(f"  → {name}" + (" (one-shot, re-triggering)" if one_shot else " (looping)") +
        ("  Ctrl-C to stop" if forever else f"  ({seconds:.0f}s)"))

  end = None if forever else time.monotonic() + seconds

  on = messaging.new_message("selfdriveState")
  on.selfdriveState.alertSound = getattr(AA, name)
  on.selfdriveState.alertText1 = name
  on.selfdriveState.enabled = True

  off = messaging.new_message("selfdriveState")
  off.selfdriveState.alertSound = AA.none
  off.selfdriveState.enabled = True

  if one_shot:
    # Approximate wav durations in seconds — used only to re-trigger play_count=1 sounds.
    DUR = {"engage": 2.05, "disengage": 2.05, "refuse": 2.05,
           "prompt": 0.7, "laneChange": 2.05}
    period = DUR.get(name, 1.0)
    while forever or time.monotonic() < end:
      # send "off" briefly so soundd sees a transition and re-fires
      for _ in range(4):
        pm.send("selfdriveState", off)
        time.sleep(0.02)
      # then "on" for one full wav play
      t_on_end = time.monotonic() + period
      while time.monotonic() < t_on_end:
        pm.send("selfdriveState", on)
        time.sleep(0.05)
  else:
    while forever or time.monotonic() < end:
      pm.send("selfdriveState", on)
      time.sleep(0.05)


def main():
  args = sys.argv[1:]
  seconds: float | None = 5.0
  if "--seconds" in args:
    i = args.index("--seconds")
    seconds = float(args[i + 1])
    args = args[:i] + args[i + 2:]

  names = args or ALL
  if len(names) == 1 and "--seconds" not in sys.argv:
    seconds = None  # single sound → loop forever

  pm = messaging.PubMaster(["selfdriveState"])
  time.sleep(0.5)  # let subscribers connect

  try:
    for n in names:
      keep_playing(n, pm, seconds)
      off = messaging.new_message("selfdriveState")
      off.selfdriveState.alertSound = AA.none
      pm.send("selfdriveState", off)
      time.sleep(0.5)
  except KeyboardInterrupt:
    off = messaging.new_message("selfdriveState")
    off.selfdriveState.alertSound = AA.none
    pm.send("selfdriveState", off)
    print("\nstopped")


if __name__ == "__main__":
  main()
