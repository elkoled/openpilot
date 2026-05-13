#!/usr/bin/env python3
# Push to the comma device, run directly — no openpilot stack, no msgq needed.
# Drives the real Soundd.get_sound_data() (which contains the loop-wrap fix)
# through sounddevice, bypassing cereal messaging.
#
#   scp selfdrive/ui/tests/play_alert_sounds.py comma@<device>:/data/openpilot/
#   ssh comma@<device>
#   # stop the openpilot stack first so it doesn't grab the audio device
#   tmux kill-session -t comma
#   cd /data/openpilot
#   python3 selfdrive/ui/tests/play_alert_sounds.py                # cycle all, 5s each
#   python3 selfdrive/ui/tests/play_alert_sounds.py laneChange      # one sound, loops until Ctrl-C
#   python3 selfdrive/ui/tests/play_alert_sounds.py --seconds 30    # cycle all, 30s each
#   python3 selfdrive/ui/tests/play_alert_sounds.py promptDistracted warningSoft --seconds 20

import sys
import time
import wave

import numpy as np
import sounddevice as sd

# Match soundd.py's loader exactly — no cereal/messaging dependency.
SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096
SOUNDS_DIR = "selfdrive/assets/sounds"

# (filename, play_count) — keep in sync with soundd.sound_list
SOUNDS = {
  "engage":            ("engage.wav",            1),
  "disengage":         ("disengage.wav",         1),
  "refuse":            ("refuse.wav",            1),
  "prompt":            ("prompt.wav",            1),
  "promptRepeat":      ("prompt.wav",            None),
  "promptDistracted":  ("prompt_distracted.wav", None),
  "warningSoft":       ("warning_soft.wav",      None),
  "warningImmediate":  ("warning_immediate.wav", None),
  "laneChange":        ("lane_change.wav",       1),
}


def load(filename: str) -> np.ndarray:
  with wave.open(f"{SOUNDS_DIR}/{filename}", "r") as w:
    assert w.getnchannels() == 1, f"{filename}: not mono"
    assert w.getsampwidth() == 2, f"{filename}: not 16-bit"
    assert w.getframerate() == SAMPLE_RATE, f"{filename}: not 48kHz"
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / (2 ** 16 / 2)


class Player:
  """Replicates Soundd.get_sound_data with the loop-wrap fix."""

  def __init__(self):
    self.loaded = {name: load(f) for name, (f, _) in SOUNDS.items()}
    self.current_alert = None
    self.current_sound_frame = 0
    self.num_loops = 1

  def set(self, name: str | None):
    self.current_alert = name
    self.current_sound_frame = 0
    self.num_loops = SOUNDS[name][1] if name else 1

  def callback(self, outdata, frames, _time, _status):
    out = np.zeros(frames, dtype=np.float32)
    if self.current_alert is not None:
      data = self.loaded[self.current_alert]
      n = len(data)
      written = 0
      while written < frames:
        loops = self.current_sound_frame // n
        if self.num_loops is not None and loops >= self.num_loops:
          break
        cur = self.current_sound_frame % n
        avail = n - cur
        ftw = min(avail, frames - written)
        out[written:written + ftw] = data[cur:cur + ftw]
        written += ftw
        self.current_sound_frame += ftw
    outdata[:, 0] = out


def main():
  args = sys.argv[1:]
  seconds: float | None = 5.0
  if "--seconds" in args:
    i = args.index("--seconds")
    seconds = float(args[i + 1])
    args = args[:i] + args[i + 2:]

  names = args or list(SOUNDS.keys())
  for n in names:
    if n not in SOUNDS:
      print(f"!! unknown sound: {n} (known: {', '.join(SOUNDS)})")
      return

  # single sound and user didn't ask for a duration → loop forever
  if len(names) == 1 and "--seconds" not in sys.argv:
    seconds = None

  p = Player()
  stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32",
                           blocksize=SAMPLE_BUFFER, callback=p.callback)
  stream.start()
  try:
    for n in names:
      forever = seconds is None
      print(f"→ {n}" + ("  (Ctrl-C to stop)" if forever else f"  ({seconds:.0f}s)"))
      p.set(n)
      end = None if forever else time.monotonic() + seconds
      while forever or time.monotonic() < end:
        time.sleep(0.05)
      p.set(None)
      time.sleep(0.5)
  except KeyboardInterrupt:
    p.set(None)
    print("\nstopped")
  finally:
    stream.stop()
    stream.close()


if __name__ == "__main__":
  main()
