#!/usr/bin/env python3
"""VW MEB radar birdseye visualizer.

Plots front-radar (Strukturen_01) objects in a top-down view, with the ego car
at origin facing +X. Uses only signals that have been verified against the
in-tree dbc/vw_meb.dbc layout — no guessed values.

Usage:
  python3 tools/vw_radar_birdseye.py <route> [--seek <seconds>] [--rate <hz>]

Notes on side radars (status as of the elkoled/1 investigation):
  - Strukturen_01 (bus 1, addr 0x24F, 64 bytes, 25 Hz) is the front radar
    fusion message. Layout in dbc/vw_meb.dbc — fully verified.
  - Two other 64-byte 25 Hz messages exist on bus 2 (0x183 and 0x234), but
    decoding them with Strukturen_01's bit layout produces saturated /
    constant values, so their layout is different and not aligned with the
    front radar. They are likely the corner radars' raw object lists in a
    proprietary format. Without a ground-truth pass (driving alongside a
    vehicle of known position to correlate), no signal definitions can be
    added without guessing — left out of dbc/ deliberately.
"""
from __future__ import annotations
import argparse
import math
import sys
import time

from opendbc.can import CANParser
from openpilot.tools.lib.logreader import LogReader

DBC = "vw_meb"
RADAR_BUS = 2

SAME, LEFT, RIGHT = "Same_Lane", "Left_Lane", "Right_Lane"
LANES = (SAME, LEFT, RIGHT)
INDICES = (1, 2)

OBJECT_KEYS = [
  (lane, idx) for lane in LANES for idx in INDICES
]

LANE_COLORS = {SAME: "#4caf50", LEFT: "#2196f3", RIGHT: "#ff9800"}


def signal_names() -> list[str]:
  out = []
  for lane in LANES:
    for idx in INDICES:
      prefix = f"{lane}_{idx:02d}"
      out += [f"{prefix}_ObjectID", f"{prefix}_Long_Distance",
              f"{prefix}_Lat_Distance", f"{prefix}_Rel_Velo"]
  return out


def collect_objects(vl) -> list[dict]:
  rows = []
  for lane, idx in OBJECT_KEYS:
    prefix = f"{lane}_{idx:02d}"
    obj_id = int(vl.get(f"{prefix}_ObjectID", 0))
    if obj_id == 0:
      continue
    rows.append({
      "lane": lane,
      "idx": idx,
      "id": obj_id,
      "long": float(vl.get(f"{prefix}_Long_Distance", 0.0)),
      "lat":  float(vl.get(f"{prefix}_Lat_Distance", 0.0)),
      "vel":  float(vl.get(f"{prefix}_Rel_Velo", 0.0)),
    })
  return rows


def render_ascii(objects: list[dict], t: float, width: int = 80, depth: int = 25) -> str:
  # X axis: lateral [-20, +20] m (right positive)
  # Y axis: longitudinal [0, 100] m (forward up)
  X_MIN, X_MAX = -20.0, 20.0
  Y_MIN, Y_MAX = 0.0, 100.0

  grid = [[" "] * width for _ in range(depth)]

  def to_xy(lat: float, lng: float):
    if not (X_MIN <= lat <= X_MAX) or not (Y_MIN <= lng <= Y_MAX):
      return None
    col = int((lat - X_MIN) / (X_MAX - X_MIN) * (width - 1))
    row = depth - 1 - int((lng - Y_MIN) / (Y_MAX - Y_MIN) * (depth - 1))
    return row, col

  ego = to_xy(0.0, 0.0)
  if ego is not None:
    r, c = ego
    grid[r][c] = "O"

  for obj in objects:
    p = to_xy(obj["lat"], obj["long"])
    if p is None:
      continue
    r, c = p
    glyph = obj["lane"][0]  # S / L / R
    grid[r][c] = glyph

  border_top = "+" + "-" * width + "+"
  body = "\n".join("|" + "".join(row) + "|" for row in grid)
  legend = "  S=Same Lane  L=Left Lane  R=Right Lane  O=Ego"

  status = f"  t={t:6.2f}s  objects={len(objects)}"
  for o in objects[:6]:
    status += f"\n    id={o['id']:3d} {o['lane']:11s} long={o['long']:6.1f}m lat={o['lat']:+6.1f}m vel={o['vel']:+5.1f}m/s"
  return f"{border_top}\n{body}\n{border_top}\n{legend}\n{status}"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("route", help="route name, e.g. aebd8f1d4ea16066/00000009--b31e222338")
  ap.add_argument("--seek", type=float, default=0.0, help="seek to N seconds before plotting")
  ap.add_argument("--rate", type=float, default=4.0, help="frames per second to print (default 4)")
  ap.add_argument("--end", type=float, default=None, help="stop after N seconds of route time")
  args = ap.parse_args()

  parser = CANParser(DBC, [("Strukturen_01", 0)], RADAR_BUS)
  msgs = LogReader(args.route)
  t0 = None
  next_print = 0.0
  delay = 1.0 / args.rate

  for m in msgs:
    if m.which() != "can":
      continue
    if t0 is None:
      t0 = m.logMonoTime
    t = (m.logMonoTime - t0) / 1e9
    if t < args.seek:
      # still feed parser so state is current when we start plotting
      parser.update([(int(m.logMonoTime), [(c.address, bytes(c.dat), c.src) for c in m.can])])
      continue
    if args.end is not None and t > args.end:
      break

    parser.update([(int(m.logMonoTime), [(c.address, bytes(c.dat), c.src) for c in m.can])])

    if t >= next_print:
      vl = parser.vl.get("Strukturen_01")
      if vl is not None:
        objects = collect_objects(vl)
        print("\033[2J\033[H" + render_ascii(objects, t), flush=True)
        time.sleep(delay)
      next_print = t + delay

  return 0


if __name__ == "__main__":
  sys.exit(main())
