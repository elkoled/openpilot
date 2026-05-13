#!/usr/bin/env python3
# Push to the comma device, stop the openpilot stack (keep soundd alive or run it
# manually), then run this to cycle every AudibleAlert through the real soundd
# pipeline via cereal messaging.
#
#   scp selfdrive/ui/tests/play_alert_sounds.py comma@<device>:/data/openpilot/
#   ssh comma@<device>
#   tmux kill-session -t comma                          # stop the stack
#   cd /data/openpilot && ./selfdrive/ui/soundd.py &    # soundd alone
#   python3 play_alert_sounds.py                        # cycle all sounds (5s each)
#   python3 play_alert_sounds.py laneChange             # single sound, loops until Ctrl-C
#   python3 play_alert_sounds.py laneChange engage      # subset (5s each)
#   python3 play_alert_sounds.py --seconds 30           # cycle all, 30s each (good for hearing loop glitches)

import sys
import time

from cereal import messaging, car

AA = car.CarControl.HUDControl.AudibleAlert

ALL = ["engage", "disengage", "refuse", "prompt", "promptRepeat",
       "promptDistracted", "warningSoft", "warningImmediate", "laneChange"]


def play(name: str, pm: messaging.PubMaster, seconds: float | None) -> None:
  """Publish alertSound=<name> until `seconds` elapses, or forever if seconds is None."""
  if not hasattr(AA, name):
    print(f"  !! {name} not in cereal schema — stale build on device?")
    return
  forever = seconds is None
  print(f"  → {name}" + ("  (Ctrl-C to stop)" if forever else f"  ({seconds:.0f}s)"))
  msg = messaging.new_message("selfdriveState")
  msg.selfdriveState.alertSound = getattr(AA, name)
  msg.selfdriveState.alertText1 = name
  msg.selfdriveState.enabled = True
  end = None if forever else time.monotonic() + seconds
  while forever or time.monotonic() < end:
    pm.send("selfdriveState", msg)
    time.sleep(0.05)


def main():
  args = sys.argv[1:]
  seconds: float | None = 5.0
  if "--seconds" in args:
    i = args.index("--seconds")
    seconds = float(args[i + 1])
    args = args[:i] + args[i + 2:]

  names = args or ALL
  # If exactly one sound is requested, loop it forever so you can listen for glitches
  if len(names) == 1 and seconds == 5.0 and "--seconds" not in sys.argv:
    seconds = None

  pm = messaging.PubMaster(["selfdriveState"])
  time.sleep(0.5)  # let subscribers connect

  try:
    for n in names:
      play(n, pm, seconds)
      # silence between sounds
      msg = messaging.new_message("selfdriveState")
      msg.selfdriveState.alertSound = AA.none
      pm.send("selfdriveState", msg)
      time.sleep(0.8)
  except KeyboardInterrupt:
    msg = messaging.new_message("selfdriveState")
    msg.selfdriveState.alertSound = AA.none
    pm.send("selfdriveState", msg)
    print("\nstopped")


if __name__ == "__main__":
  main()
