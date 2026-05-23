"""Per-dongle K(speed) slope test. Reproduces the slope table in FINDINGS.md.

Pure descriptive script; no thresholds. Reports linear fit slope of K vs
speed_bin (km/h) per dongle for the plant_fit (HCA -> yaw) and eps_fit
(HCA -> QFK) residual surfaces.
"""
from __future__ import annotations

import glob
import pickle
from collections import defaultdict

import numpy as np


def main():
  files = sorted(glob.glob('tools/lateral_maneuvers/fleet_residuals/out/*.pkl'))
  fits = defaultdict(list)  # dongle -> [(bin, K_plant, K_eps), ...]
  for f in files:
    try:
      s = pickle.load(open(f, 'rb'))
    except Exception:
      continue
    if 'error' in s:
      continue
    d = f.split('/')[-1][:16]
    for run in s.get('plant_fits', []) or []:
      sb = run.get('speed_bin_kmh')
      if sb is None:
        continue
      kp = run['plant_fit']['K'] if run.get('plant_fit') else None
      ke = run['eps_fit']['K'] if run.get('eps_fit') else None
      fits[d].append((sb, kp, ke))

  print(f"{'dongle':18s} {'plant_n':>9s} {'plant_slope':>16s} {'plant_K@40':>12s} {'plant_K@120':>13s}  "
        f"{'eps_n':>7s} {'eps_slope':>13s}")
  for d in sorted(fits):
    rows = fits[d]
    plant = [(b, kp) for b, kp, _ in rows if kp is not None]
    eps = [(b, ke) for b, _, ke in rows if ke is not None]

    if len(plant) >= 3:
      bs = np.array([b for b, _ in plant], float)
      ks = np.array([k for _, k in plant], float)
      slope, intercept = np.polyfit(bs, ks, 1)
      p40 = intercept + slope * 40
      p120 = intercept + slope * 120
      ps_str = f'{slope * 100:+.3f}/100km/h'
      p40s = f'{p40:.3f}'
      p120s = f'{p120:.3f}'
    else:
      ps_str = p40s = p120s = '   -'

    if len(eps) >= 3:
      bs = np.array([b for b, _ in eps], float)
      ks = np.array([k for _, k in eps], float)
      eslope, _ = np.polyfit(bs, ks, 1)
      es_str = f'{eslope * 100:+.3f}/100km/h'
    else:
      es_str = '   -'

    print(f'  {d}  {len(plant):>9d} {ps_str:>14s} {p40s:>12s} {p120s:>13s}  {len(eps):>7d} {es_str:>11s}')


if __name__ == '__main__':
  main()
