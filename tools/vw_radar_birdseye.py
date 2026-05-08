#!/usr/bin/env python3
"""
Bird's-eye video visualizer for VW MEB front radar (Strukturen_01) tracks.

Black background. Pink rectangle = ego. White squares = front radar tracks.

Usage:
    PYTHONPATH=/home/batman/openpilot4 uv run python tools/vw_radar_birdseye.py \\
        aebd8f1d4ea16066/00000009--b31e222338 --max-segments 4 --out /tmp/radar.mp4

NOTE: Side-radar (MEB_Side_Assist_02 / 0x24D) tracks are NOT shown here.
The 0x24D message has 64 bytes of complex content and the in-the-wild byte
pattern doesn't match any single-shot decode hypothesis we've validated end
to end. A reliable decode requires structured ground-truth recordings
(known objects at known positions) -- punted for now.
"""

import argparse
import sys

import capnp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless render -- no GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from opendbc.can import CANParser
from openpilot.tools.lib.logreader import LogReader

EMPTY_OFFSETS = np.empty((0, 2))

LANE_TYPES = ("Same_Lane", "Left_Lane", "Right_Lane")
FRONT_SIGNAL_SETS = tuple(
    (f"{prefix}_ObjectID", f"{prefix}_Long_Distance",
     f"{prefix}_Lat_Distance", f"{prefix}_Rel_Velo")
    for lane in LANE_TYPES
    for idx in (1, 2)
    for prefix in (f"{lane}_0{idx}",)
)

# Front radar (ARS) is at the front bumper. Strukturen_01 reports dRel from
# the bumper, so shift by ego half-length to plot from ego center.
EGO_HALF_LEN = 2.29  # half of ID.4 length 4.58 m
FRONT_RADAR_OFFSET_Y = EGO_HALF_LEN


def collect_frames(route, max_segments):
  cp = CANParser("vw_meb", [("Strukturen_01", 25)], 2)

  frames = []
  ego_v = 0.0

  for seg in range(max_segments):
    try:
      lr = LogReader(f"{route}/{seg}")
    except Exception as e:
      if seg == 0:
        raise
      print(f"  segment {seg}: stop ({e})")
      break

    n = 0
    for msg in lr:
      try:
        w = msg.which()
      except (capnp.lib.capnp.KjException, RuntimeError):
        continue

      if w == "carState":
        ego_v = msg.carState.vEgo
        continue
      if w != "can":
        continue

      updated = cp.update([(msg.logMonoTime,
                             [(c.address, c.dat, c.src) for c in msg.can])])
      if 0x24F not in updated:
        continue  # snapshot on front-radar trigger only

      front = cp.vl["Strukturen_01"]
      front_pts = []
      for oid_sig, long_sig, lat_sig, vel_sig in FRONT_SIGNAL_SETS:
        if int(front[oid_sig]) == 0:
          continue
        front_pts.append((float(front[long_sig]),
                           float(front[lat_sig]),
                           float(front[vel_sig])))

      frames.append((msg.logMonoTime, ego_v, front_pts))
      n += 1

    print(f"  segment {seg}: {n} radar frames")
    if n == 0 and frames:
      break

  return frames


def render_video(frames, out_path, fps=15, decimate=2):
  decimated = frames[::decimate]
  if not decimated:
    print("no frames to render")
    return

  fig = plt.figure(figsize=(6, 9), facecolor="black")
  ax = fig.add_axes([0, 0, 1, 1])
  ax.set_facecolor("black")
  ax.set_xlim(-15, 15)
  ax.set_ylim(-5, 80)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)

  ego = mpatches.Rectangle(
    (-0.92, -2.29), 1.85, 4.58,
    linewidth=1.5, edgecolor="#ff3399", facecolor="none", zorder=10,
  )
  ax.add_patch(ego)

  front_scat = ax.scatter([], [], c="white", marker="s", s=90, zorder=8)

  hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="#999999",
                fontsize=9, family="monospace", va="top")

  def animate(i):
    _, ego_v, front_pts = decimated[i]

    if front_pts:
      front_scat.set_offsets([(-p[1], p[0] + FRONT_RADAR_OFFSET_Y) for p in front_pts])
    else:
      front_scat.set_offsets(EMPTY_OFFSETS)

    hud.set_text(f"{ego_v * 3.6:5.1f} km/h   front:{len(front_pts):2d}")
    return [front_scat, hud]

  print(f"\nRendering {len(decimated)} frames at {fps} fps -> {out_path}")
  ani = FuncAnimation(fig, animate, frames=len(decimated),
                      interval=1000 / fps, blit=False)
  writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2500)
  ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "black"})
  plt.close(fig)
  print(f"  done: {out_path}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("route")
  p.add_argument("--max-segments", type=int, default=4)
  p.add_argument("--out", default="/tmp/radar_birdseye.mp4")
  p.add_argument("--fps", type=int, default=15)
  p.add_argument("--decimate", type=int, default=2,
                 help="Use every Nth radar frame (1=all, default 2)")
  p.add_argument("--max-frames", type=int, default=0,
                 help="Cap input frames before decimation (0=no cap)")
  args = p.parse_args()

  print(f"Loading route: {args.route} (up to {args.max_segments} segments)")
  frames = collect_frames(args.route, args.max_segments)
  print(f"Total radar frames: {len(frames)}")
  if not frames:
    return 1

  if args.max_frames and len(frames) > args.max_frames:
    mid = len(frames) // 2
    half = args.max_frames // 2
    frames = frames[max(0, mid - half):mid + half]
    print(f"Capped to {len(frames)} frames (middle of route)")

  render_video(frames, args.out, fps=args.fps, decimate=args.decimate)
  return 0


if __name__ == "__main__":
  sys.exit(main())
