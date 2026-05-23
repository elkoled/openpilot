#!/usr/bin/env python3
"""
Stage 1 — Manifest builder.

Probe each user-supplied route id once: qlog-read carParams (fingerprint, VIN,
steerControlType), count engaged frames, flag whether liveCurvatureParameters
was published (i.e. the dongle was running the sunnypilot dynamic_steering
learner). Tolerate every per-route failure mode (404, corrupted, missing CAN,
wrong fingerprint) without aborting the manifest.

Input file format: one route per line, `dongle_id/route_id` (or `dongle|route`).
Blank lines and `#` comments ignored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from openpilot.tools.lib.logreader import LogReader, ReadMode

TARGET_FINGERPRINT = "VOLKSWAGEN_ID4_MK1"
MIN_ENGAGED_FRAMES = 6000  # ~60 s at 100 Hz carControl; qlogs are ~2 Hz so this gets scaled below
MIN_ENGAGED_FRAMES_QLOG = 120  # qlog carControl is ~2 Hz → 60 s ≈ 120 frames; tolerate sparse
PER_ROUTE_TIMEOUT_S = 90.0


@dataclass
class ManifestRow:
  route: str
  dongle: str
  accessible: bool
  reason: str = ""
  fingerprint: str = ""
  vin: str = ""
  build_year: str = ""
  steer_control_type: str = ""
  engaged_frames_q: int = 0
  has_live_curvature_parameters: bool = False
  qlog_seconds: float = 0.0


def parse_route_id(line: str) -> str | None:
  s = line.strip()
  if not s or s.startswith("#"):
    return None
  s = s.replace("|", "/")
  if s.count("/") != 1:
    return None
  return s


def _build_year_from_vin(vin: str) -> str:
  """VIN year code is position 10 (index 9). Returns "" if not decodable."""
  if not vin or len(vin) < 10:
    return ""
  # Compact subset of VIN year code (1980-2039). Letters I, O, Q, U, Z, 0 are excluded.
  table = {
    "A": "2010", "B": "2011", "C": "2012", "D": "2013", "E": "2014",
    "F": "2015", "G": "2016", "H": "2017", "J": "2018", "K": "2019",
    "L": "2020", "M": "2021", "N": "2022", "P": "2023", "R": "2024",
    "S": "2025", "T": "2026", "V": "2027", "W": "2028", "X": "2029",
    "Y": "2030",
    "1": "2031", "2": "2032", "3": "2033", "4": "2034", "5": "2035",
    "6": "2036", "7": "2037", "8": "2038", "9": "2039",
  }
  return table.get(vin[9].upper(), "")


def probe_route(route: str) -> ManifestRow:
  dongle = route.split("/")[0]
  row = ManifestRow(route=route, dongle=dongle, accessible=False)
  t0 = time.time()
  try:
    lr = LogReader(f"{route}/q", default_mode=ReadMode.QLOG)
    fingerprint = None
    vin = ""
    steer_control_type = ""
    engaged = 0
    has_curvature = False
    last_t = 0.0
    first_t = None
    for msg in lr:
      if time.time() - t0 > PER_ROUTE_TIMEOUT_S:
        row.reason = "timeout"
        return row
      which = msg.which()
      t = msg.logMonoTime * 1e-9
      if first_t is None:
        first_t = t
      last_t = t
      if which == "carParams":
        fingerprint = msg.carParams.carFingerprint
        steer_control_type = str(msg.carParams.steerControlType)
        # VIN is a Text/String field in cereal car.capnp
        try:
          vin = msg.carParams.carVin or ""
        except Exception:
          vin = ""
      elif which == "carControl":
        if msg.carControl.latActive:
          engaged += 1
      elif which == "liveCurvatureParameters":
        has_curvature = True

    row.qlog_seconds = float((last_t - first_t) if first_t is not None else 0.0)
    row.fingerprint = fingerprint or ""
    row.vin = vin
    row.build_year = _build_year_from_vin(vin)
    row.steer_control_type = steer_control_type
    row.engaged_frames_q = engaged
    row.has_live_curvature_parameters = has_curvature

    if not fingerprint:
      row.reason = "no_car_params"
    elif fingerprint != TARGET_FINGERPRINT:
      row.reason = f"wrong_fingerprint:{fingerprint}"
    elif engaged < MIN_ENGAGED_FRAMES_QLOG:
      row.reason = f"insufficient_engaged:{engaged}"
    else:
      row.accessible = True
    return row
  except Exception as e:
    row.reason = f"exception:{type(e).__name__}:{str(e)[:120]}"
    return row


def build_manifest(route_file: str, out_path: str, num_workers: int) -> dict:
  with open(route_file) as f:
    routes = [r for r in (parse_route_id(line) for line in f) if r is not None]
  print(f"[manifest] {len(routes)} routes to probe with {num_workers} workers", flush=True)

  reasons: dict[str, int] = {}
  accessible = 0
  written = 0
  with open(out_path, "w") as out:
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
      futures = {ex.submit(probe_route, r): r for r in routes}
      for i, fut in enumerate(as_completed(futures)):
        try:
          row = fut.result()
        except Exception as e:
          route = futures[fut]
          row = ManifestRow(route=route, dongle=route.split("/")[0], accessible=False,
                            reason=f"future_exception:{type(e).__name__}")
        out.write(json.dumps(asdict(row)) + "\n")
        written += 1
        if row.accessible:
          accessible += 1
        else:
          reasons[row.reason.split(":")[0]] = reasons.get(row.reason.split(":")[0], 0) + 1
        if (i + 1) % 50 == 0:
          print(f"[manifest] {i+1}/{len(routes)} probed, {accessible} accessible", flush=True)
  summary = {
    "total": len(routes),
    "written": written,
    "accessible": accessible,
    "rejection_reasons": reasons,
  }
  print(f"[manifest] done: {json.dumps(summary)}", flush=True)
  return summary


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("route_file", help="Text file with one dongle_id/route_id per line")
  p.add_argument("--out", default="manifest.jsonl", help="Output JSONL path")
  p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
  args = p.parse_args()
  build_manifest(args.route_file, args.out, args.workers)


if __name__ == "__main__":
  sys.exit(main())
