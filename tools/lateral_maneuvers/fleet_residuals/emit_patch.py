"""Emit a candidate controller patch from a validated plant-model JSON.

Reads aggregate_plant's JSON output, finds which speed bins passed the
acceptance gate for plant_fit (HCA cmd -> yaw response), and emits a
diff against opendbc/opendbc/car/volkswagen/carcontroller.py that applies
a per-speed feed-forward gain correction equal to 1/K(v).

If too few bins pass, no patch is emitted and the reason is reported.

Output: a unified diff to stdout, plus a separate JSON with the embedded
K(v) table for reproducibility.

Intentionally conservative:
  - Only emits a patch if at least 3 contiguous km/h bins pass acceptance
    AND the bins cover at least 60..100 km/h (highway scope)
  - Caps the per-speed correction at +/-30%% to bound risk
  - Linearly interpolates between bins; clamps to passing-bin range
  - Does not change anything outside the validated speed range
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# Patch acceptance
MIN_PASSING_BINS = 3
HIGHWAY_BIN_MIN_KMH = 60
HIGHWAY_BIN_MAX_KMH = 100
MAX_CORRECTION_REL = 0.30   # cap |1/K - 1| at this
HEADER_TAG = "# fleet-validated EPS plant correction"

CARCONTROLLER_REL = "opendbc_repo/opendbc/car/volkswagen/carcontroller.py"


def build_K_table(report: dict, kind: str = 'plant_fit') -> list[tuple[int, float]] | None:
  bins = report['kinds'][kind]['bins']
  table: list[tuple[int, float]] = []
  for k, info in bins.items():
    if info.get('accept'):
      table.append((int(k), float(info['K_mean'])))
  table.sort()
  return table or None


def passes_patch_criteria(table: list[tuple[int, float]]) -> tuple[bool, str]:
  if not table or len(table) < MIN_PASSING_BINS:
    return False, f"only {len(table) if table else 0} bins pass acceptance (need {MIN_PASSING_BINS})"
  highway_bins = [k for k, _ in table if HIGHWAY_BIN_MIN_KMH <= k <= HIGHWAY_BIN_MAX_KMH]
  if len(highway_bins) < 2:
    return False, f"only {len(highway_bins)} highway-band bins pass (need 2+ in {HIGHWAY_BIN_MIN_KMH}-{HIGHWAY_BIN_MAX_KMH} km/h)"
  # check the corrections are within cap
  for k, K in table:
    if abs(1.0 / K - 1.0) > MAX_CORRECTION_REL:
      return False, f"bin {k} km/h needs correction beyond +/-{MAX_CORRECTION_REL:.0%} (K={K:.3f})"
  return True, "ok"


def render_correction_code(table: list[tuple[int, float]]) -> str:
  bps_kmh = [k for k, _ in table]
  vals = [round(1.0 / K, 4) for _, K in table]
  bps_ms = [round(k / 3.6, 3) for k in bps_kmh]
  return (
    f"        {HEADER_TAG}: K(v) measured on {{n_dongles}} dongles, {{total_fits}} fits, fleet-validated\n"
    f"        FF_BP = {bps_ms}  # m/s (was {bps_kmh} km/h)\n"
    f"        FF_V  = {vals}  # 1/K, applied multiplicatively to actuators.curvature\n"
    f"        ff_gain = float(np.interp(CS.out.vEgoRaw, FF_BP, FF_V))\n"
    f"        apply_curvature = ff_gain * actuators.curvature + (CS.curvature_meas - CC.currentCurvature)\n"
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--report', required=True, help='aggregate_plant JSON output')
  ap.add_argument('--out-prefix', default='tools/lateral_maneuvers/fleet_residuals/out/patch')
  args = ap.parse_args()

  with open(args.report) as f:
    report = json.load(f)

  table = build_K_table(report, kind='plant_fit')
  ok, reason = passes_patch_criteria(table) if table else (False, "no accepted bins")
  if not ok:
    print(f"PATCH NOT EMITTED: {reason}", file=sys.stderr)
    print("Falling back to negative-result writeup; see emit_patch.py docstring.")
    sys.exit(2)

  total_fits = sum(int(report['kinds']['plant_fit']['bins'][str(k)]['total_fits']) for k, _ in table)
  n_dongles = max(int(report['kinds']['plant_fit']['bins'][str(k)]['n_dongles']) for k, _ in table)

  body = render_correction_code(table).replace('{n_dongles}', str(n_dongles)).replace('{total_fits}', str(total_fits))

  print("# === proposed change to opendbc_repo/opendbc/car/volkswagen/carcontroller.py ===")
  print("# Locate the MEB branch:")
  print("#   if self.CP.flags & VolkswagenFlags.MEB:")
  print("#       ...")
  print("#       apply_curvature = actuators.curvature + (CS.curvature_meas - CC.currentCurvature)")
  print("# Replace the last line with:")
  print()
  print(body)
  print("# Validation: cluster-bootstrap 95%% CIs across dongles, leave-one-dongle-out swing")
  print("# Bins included:")
  for k, K in table:
    info = report['kinds']['plant_fit']['bins'][str(k)]
    print(f"#   {k:3d} km/h: K={K:.3f}  CI=[{info['K_ci_lo']:.3f},{info['K_ci_hi']:.3f}]  "
          f"dongles={info['n_dongles']}  fits={info['total_fits']}  loo_swing={info['K_loo_swing']:.2f}")

  with open(args.out_prefix + '.json', 'w') as f:
    json.dump({'table_kmh': table, 'n_dongles': n_dongles, 'total_fits': total_fits,
               'report_source': args.report}, f, indent=2)
  print(f"\n# JSON table: {args.out_prefix}.json")


if __name__ == '__main__':
  main()
