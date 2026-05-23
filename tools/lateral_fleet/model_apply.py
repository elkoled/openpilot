"""K(v) lookup + safe correction-curvature computation.

Given the fleet K(v) table produced by `plant_fit.fit_all`, this module
exposes:

  - load_fleet_K(run_dir, kind='yaw') -> array of (v_kph, K_mean, K_lo, K_hi)
  - apply_correction(c_cmd, v_ego, K_table, max_boost=2.0) -> c_cmd_corrected

The correction is **gain inversion**: c_cmd_out = c_cmd_in / K(v). If the
EPS undershoots by K=0.7 (i.e. commands 1.0 result in 0.7 of actual), we
boost the command by 1/0.7 ≈ 1.43 so the plant output matches the target.

Safety:
  - `max_boost` caps the multiplier (default 2x) — never amplify by more
    than this regardless of K.
  - K(v) is interpolated linearly between speed anchors.
  - At speeds below the lowest fitted anchor, no correction is applied
    (low-speed driving is already fine per user feedback).
  - At speeds above the highest fitted anchor, the correction is clamped
    to the value at the highest anchor.
  - If a speed bucket has fewer than `MIN_DONGLES_VALID` dongles
    contributing, the correction at that speed is 1.0 (passthrough).

This is the open-loop steady-state correction. It does NOT account for
plant lag (use `liveDelay` for that, separately) and does NOT change
panda safety limits.

USAGE in openpilot's MEB carcontroller would be:

  from openpilot.tools.lateral_fleet.model_apply import (
    load_fleet_K, apply_correction,
  )
  K_TABLE = load_fleet_K('/path/to/run/dir', kind='yaw')

  # In carcontroller.update():
  c_cmd = actuators.curvature + (CS.curvature_meas - CC.currentCurvature)
  c_cmd = apply_correction(c_cmd, CS.vEgoRaw, K_TABLE)
  c_cmd = apply_std_curvature_limits(c_cmd, ...)

But: do not ship this without closed-loop testing. The plant has lag, and
inverting gain in an open-loop sense can cause limit-cycle oscillations
if the rate limit is hit. Test on a single car first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_DONGLES_VALID = 5
MAX_BOOST_DEFAULT = 2.0


def load_fleet_K(run_dir: Path | str, kind: str = 'yaw') -> np.ndarray:
  """Return a (N, 4) array of [v_kph, K_mean, K_lo, K_hi], NaN for
  insufficient buckets. Sorted by v_kph."""
  p = Path(run_dir) / f'plant_fleet_{kind}.parquet'
  df = pd.read_parquet(p).sort_values('speed_kph').reset_index(drop=True)
  out = np.column_stack([
    df['speed_kph'].to_numpy(),
    df['K_fleet_mean'].to_numpy(),
    df['K_fleet_lo'].to_numpy(),
    df['K_fleet_hi'].to_numpy(),
  ])
  # Mask out cells with too-few dongles.
  invalid = df['n_dongles_valid'].to_numpy() < MIN_DONGLES_VALID
  out[invalid, 1:] = np.nan
  return out


def K_at_speed(v_ego: float, K_table: np.ndarray) -> float:
  """Linear-interp K_mean at v_ego (m/s); returns 1.0 for out-of-range
  or insufficient cells (passthrough)."""
  v_kph = float(v_ego) * 3.6
  speeds = K_table[:, 0]
  Ks = K_table[:, 1]
  if v_kph < speeds[0]:
    return 1.0
  if v_kph > speeds[-1]:
    last = Ks[-1]
    return float(last) if np.isfinite(last) else 1.0
  K = float(np.interp(v_kph, speeds, Ks))
  return K if np.isfinite(K) else 1.0


def apply_correction(c_cmd: float, v_ego: float, K_table: np.ndarray,
                     max_boost: float = MAX_BOOST_DEFAULT) -> float:
  """Boost commanded curvature by 1/K(v_ego), capped at `max_boost`."""
  K = K_at_speed(v_ego, K_table)
  if K <= 0 or not np.isfinite(K):
    return float(c_cmd)
  boost = min(1.0 / K, max_boost)
  return float(c_cmd) * boost


def correction_summary(K_table: np.ndarray) -> pd.DataFrame:
  """Diagnostic: per-speed K and recommended boost factor."""
  rows = []
  for v_kph, K, lo, hi in K_table:
    boost = min(1.0 / K, MAX_BOOST_DEFAULT) if K > 0 and np.isfinite(K) else 1.0
    rows.append({
      'speed_kph': v_kph, 'K_mean': K, 'K_lo': lo, 'K_hi': hi,
      'recommended_boost': boost,
      'within_max_boost': bool(np.isfinite(K) and 1.0 / max(K, 1e-9) <= MAX_BOOST_DEFAULT),
    })
  return pd.DataFrame(rows)
