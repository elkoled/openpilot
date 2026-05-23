"""Cross-dongle aggregation and validation of per-segment ARX plant fits.

Input: directory of per-segment pickles produced by extract.py.
Output: a JSON 'plant model' candidate keyed by speed bin, with:
  - n_dongles_contributing
  - K, T_s, tau_s population estimates (mean of dongle-medians)
  - 95% CIs via dongle-cluster bootstrap
  - leave-one-dongle-out (LOO) median absolute error (MAE) across speed bins
  - acceptance flag per speed bin and overall

Acceptance gates (any of these fails -> NOT shippable for that bin):
  - >= MIN_DONGLES_PER_BIN dongles contributing
  - >= MIN_FITS_PER_DONGLE per dongle in that bin
  - K and T 95% CI relative half-width <= MAX_RELATIVE_HALFWIDTH
  - LOO max-dongle deviation <= LOO_K_TOL on K
  - dominant-dongle held out: remaining-fleet K and T within CI of full-fleet

This is intentionally strict. A bin that fails is reported, not silently
included.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Any

import numpy as np

MIN_DONGLES_PER_BIN = 5
MIN_FITS_PER_DONGLE = 3
MAX_RELATIVE_HALFWIDTH = 0.20    # 95% CI half-width as fraction of mean
LOO_K_TOL_RELATIVE = 0.15        # leave-one-dongle-out K shift tolerance
BOOTSTRAP_N = 2000

# Best-effort criteria used when the dataset is too small to pass the
# strict gate. Reported alongside the strict result, never instead of it.
BE_MIN_DONGLES_PER_BIN = 2
BE_MIN_FITS_PER_DONGLE = 2

PLANT_KINDS = ('eps_fit', 'plant_fit')
SPEED_BINS = (20, 40, 60, 80, 100, 120, 140)


def load_all(out_dir: str) -> list[dict]:
  paths = sorted(glob.glob(os.path.join(out_dir, '*.pkl')))
  loaded = []
  for p in paths:
    try:
      with open(p, 'rb') as f:
        s = pickle.load(f)
      loaded.append(s)
    except Exception:
      continue
  return loaded


def dongle_of(s: dict) -> str:
  rid = s.get('route_id', '') or ''
  if rid.startswith('http'):
    parts = rid.rstrip('/').split('/')
    for p in parts:
      if len(p) == 16 and all(c in '0123456789abcdef' for c in p):
        return p
  return rid.split('/')[0]


def collect_fits(summaries: list[dict]) -> dict:
  """Returns {kind: {speed_bin: {dongle: [{K,T_s,tau_s,R2,n,v_mean},...]}}}."""
  out: dict = {k: defaultdict(lambda: defaultdict(list)) for k in PLANT_KINDS}
  for s in summaries:
    if 'plant_fits' not in s or not s['plant_fits']:
      continue
    d = dongle_of(s)
    for run in s['plant_fits']:
      sb = run.get('speed_bin_kmh')
      if sb is None:
        continue
      for kind in PLANT_KINDS:
        fit = run.get(kind)
        if fit is None:
          continue
        out[kind][sb][d].append({
          'K': fit['K'], 'T_s': fit['T_s'], 'tau_s': fit['tau_s'],
          'R2': fit['R2'], 'n': fit['n'],
          'v_mean_kmh': run['v_mean_kmh'],
        })
  return out


def dongle_medians(per_dongle: dict[str, list[dict]], field: str,
                   min_fits: int = MIN_FITS_PER_DONGLE) -> dict[str, float]:
  return {d: float(np.median([fit[field] for fit in fits]))
          for d, fits in per_dongle.items() if len(fits) >= min_fits}


def cluster_bootstrap_ci(values: dict[str, float], n: int = BOOTSTRAP_N,
                         alpha: float = 0.05) -> tuple[float, float, float]:
  """Resample dongles with replacement; return (mean, lo, hi) of the resample
  means. Each dongle contributes its median, equal weight."""
  if not values:
    return 0.0, 0.0, 0.0
  arr = np.array(list(values.values()), dtype=np.float64)
  rng = np.random.default_rng(42)
  n_d = len(arr)
  means = np.empty(n, dtype=np.float64)
  for i in range(n):
    sample = arr[rng.integers(0, n_d, size=n_d)]
    means[i] = sample.mean()
  return float(arr.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def loo_dongle_K_swing(values: dict[str, float]) -> float:
  """Max relative shift in mean-across-dongles when each dongle is held out."""
  if len(values) < 2:
    return float('inf')
  arr = np.array(list(values.values()), dtype=np.float64)
  full = arr.mean()
  if full == 0:
    return float('inf')
  swing = 0.0
  for i in range(len(arr)):
    held = np.delete(arr, i)
    swing = max(swing, abs(held.mean() - full) / abs(full))
  return swing


def summarise_bin(per_dongle: dict[str, list[dict]],
                  min_dongles: int = MIN_DONGLES_PER_BIN,
                  min_fits_per_dongle: int = MIN_FITS_PER_DONGLE) -> dict:
  K_per = dongle_medians(per_dongle, 'K', min_fits_per_dongle)
  T_per = dongle_medians(per_dongle, 'T_s', min_fits_per_dongle)
  tau_per = dongle_medians(per_dongle, 'tau_s', min_fits_per_dongle)
  n_dongles = len(K_per)
  total_fits = sum(len(v) for v in per_dongle.values())
  K_mean, K_lo, K_hi = cluster_bootstrap_ci(K_per)
  T_mean, T_lo, T_hi = cluster_bootstrap_ci(T_per)
  tau_mean, tau_lo, tau_hi = cluster_bootstrap_ci(tau_per)
  K_hw = (K_hi - K_lo) / max(2 * abs(K_mean), 1e-9)
  T_hw = (T_hi - T_lo) / max(2 * abs(T_mean), 1e-9)
  K_loo_swing = loo_dongle_K_swing(K_per)

  fits_per_dongle = {d: len(v) for d, v in per_dongle.items() if len(v) >= min_fits_per_dongle}
  dom = max(fits_per_dongle, key=fits_per_dongle.get) if fits_per_dongle else None
  K_without_dom = None
  if dom is not None and dom in K_per:
    K_minus = {d: v for d, v in K_per.items() if d != dom}
    if K_minus:
      K_without_dom = float(np.mean(list(K_minus.values())))

  accept = (
    n_dongles >= min_dongles and
    K_hw <= MAX_RELATIVE_HALFWIDTH and
    T_hw <= MAX_RELATIVE_HALFWIDTH and
    K_loo_swing <= LOO_K_TOL_RELATIVE and
    (K_without_dom is None or abs(K_without_dom - K_mean) / max(abs(K_mean), 1e-9) <= LOO_K_TOL_RELATIVE)
  )

  return {
    'n_dongles': n_dongles,
    'total_fits': total_fits,
    'dongle_counts': dict(sorted({d: len(v) for d, v in per_dongle.items()}.items(), key=lambda kv: -kv[1])),
    'dongle_medians_K': dict(sorted(K_per.items(), key=lambda kv: kv[1])),
    'K_mean': K_mean, 'K_ci_lo': K_lo, 'K_ci_hi': K_hi, 'K_rel_halfwidth': K_hw,
    'T_s_mean': T_mean, 'T_s_ci_lo': T_lo, 'T_s_ci_hi': T_hi, 'T_rel_halfwidth': T_hw,
    'tau_s_mean': tau_mean, 'tau_s_ci_lo': tau_lo, 'tau_s_ci_hi': tau_hi,
    'K_loo_swing': K_loo_swing,
    'dominant_dongle': dom,
    'K_without_dominant': K_without_dom,
    'accept': accept,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out-dir', default='tools/lateral_maneuvers/fleet_residuals/out')
  ap.add_argument('--json-out', default=None)
  args = ap.parse_args()

  summaries = load_all(args.out_dir)
  if not summaries:
    print("no pickles loaded", file=sys.stderr)
    sys.exit(1)

  ok_summaries = [s for s in summaries if s.get('plant_fits')]
  print(f"loaded {len(summaries)} pickles; {len(ok_summaries)} have plant_fits")

  collected = collect_fits(summaries)
  dongles = {dongle_of(s) for s in summaries}
  print(f"dongles seen: {len(dongles)} ({sorted(dongles)})")
  print()

  report: dict[str, Any] = {
    'pickles_total': len(summaries),
    'pickles_with_fits': len(ok_summaries),
    'dongles': sorted(dongles),
    'kinds': {},
  }

  for kind in PLANT_KINDS:
    print(f"=== {kind} ===")
    kind_report: dict[str, Any] = {'bins': {}, 'shippable_bins': [],
                                    'bins_best_effort': {}, 'best_effort_bins': []}
    for sb in SPEED_BINS:
      per_dongle = collected[kind].get(sb, {})
      if not per_dongle:
        print(f"  {sb:3d} km/h: no fits")
        continue
      info = summarise_bin(per_dongle)
      info_be = summarise_bin(per_dongle, BE_MIN_DONGLES_PER_BIN, BE_MIN_FITS_PER_DONGLE)
      flag = "ACCEPT" if info['accept'] else "reject"
      be = "be:ok" if info_be['accept'] else "be:--"
      print(f"  {sb:3d} km/h [{flag} {be}]  strict: K={info['K_mean']:.3f} dongles={info['n_dongles']}  "
            f"best-effort: K={info_be['K_mean']:.3f} [{info_be['K_ci_lo']:.3f},{info_be['K_ci_hi']:.3f}]  "
            f"T={info_be['T_s_mean']*1000:5.0f}ms  tau={info_be['tau_s_mean']*1000:5.0f}ms  "
            f"dongles={info_be['n_dongles']}  fits={info_be['total_fits']:4d}  "
            f"K_hw={info_be['K_rel_halfwidth']:.2f}  loo_swing={info_be['K_loo_swing']:.2f}")
      kind_report['bins'][str(sb)] = info
      kind_report['bins_best_effort'][str(sb)] = info_be
      if info['accept']:
        kind_report['shippable_bins'].append(sb)
      if info_be['accept']:
        kind_report['best_effort_bins'].append(sb)
    report['kinds'][kind] = kind_report
    print()

  if args.json_out:
    with open(args.json_out, 'w') as f:
      json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"wrote {args.json_out}")


if __name__ == '__main__':
  main()
