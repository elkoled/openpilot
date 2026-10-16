#!/usr/bin/env python3
# Fits the VW MEB EPS response (delay, lag, gain) per speed band from normal engaged driving, at the stock steering power.
# Prints a table and can write the baseline the onroad EPS test screen draws against the live response.
# usage: meb_eps_fit.py <route or segment range> [--baseline /data/meb_eps_baseline.json]
import argparse
import json
from collections import defaultdict

import numpy as np

from opendbc.can import CANParser
from opendbc.car import Bus
from opendbc.car.volkswagen.power_test import DT, STOCK_POWER
from opendbc.car.volkswagen.power_test_fit import fit
from opendbc.car.volkswagen.values import CAR, DBC
from openpilot.tools.lib.logreader import LogReader

BANDS = ((2., 5.), (5., 8.), (8., 12.), (12., 20.), (20., 30.), (30., 45.))  # m/s
MIN_SEGMENT = 2.  # s
MAX_PER_BAND = 300.  # s, enough for a stable fit and keeps it quick


def signed(vl: dict, name: str) -> float:
  return vl[name]["Curvature"] * (1 if vl[name]["Curvature_VZ"] else -1)


def read(route: str) -> np.ndarray:
  # one row per steering message: t, v_ego, ok to fit, curvature sent, curvature measured
  dbc = DBC[CAR.VOLKSWAGEN_ID4_MK2][Bus.pt]
  rx = CANParser(dbc, [("QFK_01", float("nan"))], 0)
  tx = CANParser(dbc, [("HCA_03", float("nan"))], 0)
  rows = []
  v_ego, pressed, lat_active = 0., True, False
  for m in LogReader(route):
    w = m.which()
    if w == "carState":
      v_ego, pressed = m.carState.vEgo, m.carState.steeringPressed
    elif w == "carControl":
      lat_active = m.carControl.latActive
    elif w == "can":
      rx.update([(m.logMonoTime, [(c.address, c.dat, c.src) for c in m.can if c.src == 0])])
    elif w == "sendcan":
      hca = [(c.address, c.dat, 0) for c in m.sendcan if c.src == 0 and c.address == 0x303]
      if hca:
        tx.update([(m.logMonoTime, hca)])
        ok = lat_active and not pressed and tx.vl["HCA_03"]["Power"] == STOCK_POWER
        rows.append((m.logMonoTime * 1e-9, v_ego, ok, signed(tx.vl, "HCA_03"), signed(rx.vl, "QFK_01")))
  return np.array(rows, dtype=float)


def segments(rows: np.ndarray) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
  t, v, ok, cmd, meas = rows.T
  band = np.full(len(t), -1)
  for i, (lo, hi) in enumerate(BANDS):
    band[(v >= lo) & (v < hi)] = i
  good = (ok > 0) & (band >= 0) & np.isfinite(cmd) & np.isfinite(meas)
  # contiguous runs in one band without gaps
  brk = np.r_[True, (band[1:] != band[:-1]) | ~good[1:] | ~good[:-1] | (np.diff(t) > 3 * DT)]
  out: dict[int, list] = defaultdict(list)
  for idx in np.split(np.arange(len(t)), np.flatnonzero(brk)):
    if len(idx) and good[idx[0]] and t[idx[-1]] - t[idx[0]] >= MIN_SEGMENT:
      tt = np.arange(t[idx[0]], t[idx[-1]], DT)
      out[int(band[idx[0]])].append((np.interp(tt, t[idx], cmd[idx]), np.interp(tt, t[idx], meas[idx])))
  # the longest runs first, up to the cap
  for b, segs in out.items():
    segs.sort(key=len, reverse=True)
    n = np.cumsum([len(c) for c, _ in segs]) * DT
    out[b] = segs[:max(int(np.searchsorted(n, MAX_PER_BAND)), 1)]
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("route")
  parser.add_argument("--baseline", help="write the fit here for the onroad EPS test screen")
  args = parser.parse_args()

  segs = segments(read(args.route))
  print(f"stock power {STOCK_POWER}%")
  print(f"{'band m/s':>9} {'delay ms':>8} {'tau s':>6} {'gain':>5} {'r2':>5} {'minutes':>7}")
  baseline = {"v": [], "delay_ms": [], "tau": [], "gain": []}
  for b, (lo, hi) in enumerate(BANDS):
    # normal driving rarely reaches the EPS rate limit, fit without one
    f = fit(segs.get(b, []), rate=np.inf)
    if f is None:
      continue
    print(f"{lo:4.0f}-{hi:<4.0f} {f['delay_ms']:8.0f} {f['tau']:6.3f} {f['gain']:5.2f} {f['r2']:5.2f} {f['n'] * DT / 60:7.1f}")
    for k, x in (("v", (lo + hi) / 2), ("delay_ms", f["delay_ms"]), ("tau", f["tau"]), ("gain", f["gain"])):
      baseline[k].append(round(float(x), 3))
  if args.baseline and baseline["v"]:
    with open(args.baseline, "w") as fh:
      json.dump(baseline, fh)
    print(f"baseline written to {args.baseline}")


if __name__ == "__main__":
  main()
