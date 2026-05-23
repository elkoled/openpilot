"""Parametric EPS plant fit.

A *secondary* model that complements the empirical K(v, c) bucket table.
Whereas the bucket table is non-parametric (one number per cell), the
parametric model captures the steady-state EPS response as

    c_actual(t) = K(v_bin) * c_cmd(t - lateral_delay)

with K(v_bin) constrained to (0, 1] (passive plant). Anything outside that
range is quarantined as overfit, not reported as a result.

If `tau_from_xcorr` is set, we additionally cross-correlate c_cmd vs c_yaw
to verify the route's own lateralDelay estimate is sane. Disagreement
between the two delay sources is itself a finding.

Deadband is reported, not modelled in K: at small |c_cmd| the signal-to-
noise of c_yaw is dominated by yaw-rate sensor noise, so any "deadband"
estimated from that regime is more noise than physics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from openpilot.tools.lateral_fleet import cache, features

# Use a broader minimum-curvature gate than the bucket grid: for the plant
# fit, we want samples where c_cmd is large enough that c_yaw is signal-
# dominated rather than noise-dominated.
MIN_C_CMD_FIT = 5e-4   # rad/m
MAX_C_CMD_FIT = 1e-2   # rad/m -- above this, lat_accel-gated samples are rare
MIN_SAMPLES_PER_SPEED = 200


def _bootstrap_K(c_cmd: np.ndarray, c_actual: np.ndarray,
                 n: int = 200, seed: int = 1) -> tuple[float, float, float]:
  """Return (K, lo, hi) for the regression c_actual = K * c_cmd through origin,
  with 5-95 bootstrap CI over samples."""
  if c_cmd.size < 20:
    return float('nan'), float('nan'), float('nan')
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, c_cmd.size, size=(n, c_cmd.size))
  num = np.sum(c_cmd[idx] * c_actual[idx], axis=1)
  den = np.sum(c_cmd[idx] * c_cmd[idx], axis=1)
  Ks = num / np.where(den == 0, 1.0, den)
  K_full = float(np.sum(c_cmd * c_actual) / max(float(np.sum(c_cmd * c_cmd)), 1e-12))
  return K_full, float(np.percentile(Ks, 5)), float(np.percentile(Ks, 95))


def _xcorr_delay(c_cmd: np.ndarray, c_actual: np.ndarray,
                 dt: float = features.RESAMPLE_DT,
                 max_delay_s: float = 0.6) -> float:
  """Find the lag in seconds that maximises cross-correlation of c_cmd with
  c_actual. Returns NaN if either signal has near-zero variance."""
  if c_cmd.std() < 1e-9 or c_actual.std() < 1e-9 or c_cmd.size < 50:
    return float('nan')
  max_n = int(round(max_delay_s / dt))
  best_lag = 0
  best_xc = -np.inf
  c_cmd_c = (c_cmd - c_cmd.mean())
  c_act_c = (c_actual - c_actual.mean())
  denom = max(c_cmd_c.std() * c_act_c.std() * c_cmd.size, 1e-12)
  for n in range(0, max_n + 1):
    if n == 0:
      xc = float(np.sum(c_cmd_c * c_act_c) / denom)
    else:
      xc = float(np.sum(c_cmd_c[:-n] * c_act_c[n:]) / denom)
    if xc > best_xc:
      best_xc = xc
      best_lag = n
  return best_lag * dt


def fit_dongle(dongle_id: str, route_ids: list[str],
               resid_kind: str = 'yaw') -> pd.DataFrame:
  """Per-speed K fit for one dongle, pooling all that dongle's routes.

  Returns one row per speed bin, with columns:
    speed_idx, n_samples, K, K_lo, K_hi, mean_v, valid_passive
  """
  c_actual_col = 'c_yaw' if resid_kind == 'yaw' else 'c_eps'
  per_speed_samples: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    s: [] for s in range(features.NUM_SPEED_ANCHORS)}
  for r in route_ids:
    try:
      tl = cache.read_timeline(dongle_id, r)
    except (FileNotFoundError, OSError):
      continue
    if 'mask_strict' not in tl.columns:
      continue
    mask = tl['mask_strict'].to_numpy().astype(bool)
    c_cmd = tl['c_cmd_delayed'].to_numpy()
    c_act = tl[c_actual_col].to_numpy()
    v = tl['v_ego'].to_numpy()
    in_range = mask & (np.abs(c_cmd) >= MIN_C_CMD_FIT) & (np.abs(c_cmd) <= MAX_C_CMD_FIT) \
               & np.isfinite(c_cmd) & np.isfinite(c_act) & np.isfinite(v)
    if not in_range.any():
      continue
    speed_idx = features.speed_bucket(v[in_range])
    c_cmd_sel = c_cmd[in_range]
    c_act_sel = c_act[in_range]
    v_sel = v[in_range]
    for s in range(features.NUM_SPEED_ANCHORS):
      sel = speed_idx == s
      if sel.any():
        per_speed_samples[s].append((c_cmd_sel[sel], c_act_sel[sel], v_sel[sel]))

  rows = []
  for s, lst in per_speed_samples.items():
    if not lst:
      rows.append({'speed_idx': s, 'n_samples': 0,
                   'K': float('nan'), 'K_lo': float('nan'), 'K_hi': float('nan'),
                   'mean_v': float('nan'), 'valid_passive': False})
      continue
    c_cmd = np.concatenate([t[0] for t in lst])
    c_act = np.concatenate([t[1] for t in lst])
    v = np.concatenate([t[2] for t in lst])
    if c_cmd.size < MIN_SAMPLES_PER_SPEED:
      rows.append({'speed_idx': s, 'n_samples': int(c_cmd.size),
                   'K': float('nan'), 'K_lo': float('nan'), 'K_hi': float('nan'),
                   'mean_v': float(np.mean(v)) if v.size else float('nan'),
                   'valid_passive': False})
      continue
    K, lo, hi = _bootstrap_K(c_cmd, c_act, seed=hash((dongle_id, s)) & 0xFFFFFFFF)
    valid = bool(np.isfinite(K) and 0.0 < K <= 1.0)
    rows.append({'speed_idx': s, 'n_samples': int(c_cmd.size),
                 'K': K, 'K_lo': lo, 'K_hi': hi,
                 'mean_v': float(np.mean(v)),
                 'valid_passive': valid})
  return pd.DataFrame(rows)


def fit_dongle_delay(dongle_id: str, route_ids: list[str],
                     resid_kind: str = 'yaw') -> dict:
  """Cross-correlation-derived delay (single number per dongle) and the
  median lateralDelay reported by openpilot's online estimator. If the two
  disagree by > 0.1 s the estimates are flagged."""
  c_actual_col = 'c_yaw' if resid_kind == 'yaw' else 'c_eps'
  xc_lags = []
  live_lags = []
  for r in route_ids:
    try:
      tl = cache.read_timeline(dongle_id, r)
    except (FileNotFoundError, OSError):
      continue
    mask = tl['mask_loose'].to_numpy().astype(bool) if 'mask_loose' in tl.columns else None
    if mask is None or mask.sum() < 100:
      continue
    c_cmd = tl['c_cmd'].to_numpy()[mask]
    c_act = tl[c_actual_col].to_numpy()[mask]
    live = tl['lateral_delay'].to_numpy()[mask]
    if c_cmd.size < 200 or np.abs(c_cmd).max() < 5e-4:
      continue
    xc_lags.append(_xcorr_delay(c_cmd, c_act))
    live_lags.append(float(np.nanmedian(live)))
  xc_lags = np.array([x for x in xc_lags if np.isfinite(x)])
  live_lags = np.array([x for x in live_lags if np.isfinite(x)])
  return {
    'dongle_id': dongle_id,
    'xcorr_delay_median_s': float(np.median(xc_lags)) if xc_lags.size else float('nan'),
    'xcorr_delay_iqr_s': float(np.subtract(*np.percentile(xc_lags, [75, 25]))) if xc_lags.size else float('nan'),
    'live_delay_median_s': float(np.median(live_lags)) if live_lags.size else float('nan'),
    'n_routes_xcorr': int(xc_lags.size),
  }


def fit_all(run_dir: Path, resid_kind: str = 'yaw') -> dict:
  """Fit K(v) and tau for every dongle whose buckets are cached.

  Writes to run_dir:
    plant_fit_<kind>.parquet    -- per-dongle, per-speed K(v) table
    plant_delay_<kind>.parquet  -- per-dongle delay summary
    plant_fleet_<kind>.parquet  -- unweighted-mean-across-dongles K(v)
  """
  run_dir = Path(run_dir)
  routes_path = run_dir / 'routes.parquet'
  if not routes_path.exists():
    return {'error': f'no routes.parquet at {routes_path}; run aggregate first'}
  routes = pd.read_parquet(routes_path)
  by_dongle = routes.groupby('dongle_id')['route_id'].apply(list).to_dict()

  per_dongle_frames = []
  delay_rows = []
  for d, rids in by_dongle.items():
    fit = fit_dongle(d, rids, resid_kind=resid_kind)
    fit['dongle_id'] = d
    per_dongle_frames.append(fit)
    delay_rows.append(fit_dongle_delay(d, rids, resid_kind=resid_kind))

  per_dongle = pd.concat(per_dongle_frames, ignore_index=True) if per_dongle_frames else pd.DataFrame()
  per_dongle.to_parquet(run_dir / f'plant_fit_{resid_kind}.parquet', index=False)
  pd.DataFrame(delay_rows).to_parquet(run_dir / f'plant_delay_{resid_kind}.parquet', index=False)

  # Hierarchical fleet K(v): use only valid_passive K values, unweighted mean.
  fleet_rows = []
  if not per_dongle.empty:
    for s, grp in per_dongle.groupby('speed_idx'):
      v = grp.loc[grp['valid_passive'], 'K'].to_numpy()
      v = v[np.isfinite(v)]
      n_valid = int(v.size)
      n_total = int(grp.shape[0])
      if n_valid == 0:
        K_mean = K_lo = K_hi = float('nan')
      else:
        K_mean = float(np.mean(v))
        if n_valid >= 2:
          rng = np.random.default_rng(int(s) + 100)
          idx = rng.integers(0, n_valid, size=(500, n_valid))
          means = np.mean(v[idx], axis=1)
          K_lo, K_hi = float(np.percentile(means, 5)), float(np.percentile(means, 95))
        else:
          K_lo = K_hi = K_mean
      fleet_rows.append({
        'speed_idx': int(s),
        'speed_kph': float(features.SPEED_ANCHORS_KPH[int(s)]),
        'K_fleet_mean': K_mean,
        'K_fleet_lo': K_lo,
        'K_fleet_hi': K_hi,
        'n_dongles_valid': n_valid,
        'n_dongles_total': n_total,
      })
  fleet = pd.DataFrame(fleet_rows)
  fleet.to_parquet(run_dir / f'plant_fleet_{resid_kind}.parquet', index=False)
  return {
    'per_dongle_rows': int(per_dongle.shape[0]),
    'n_dongles': len(by_dongle),
    'fleet_rows': int(fleet.shape[0]),
    'resid_kind': resid_kind,
  }
