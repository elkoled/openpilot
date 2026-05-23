"""Per-dongle + cross-dongle aggregation.

Routes (or segments) are the unit of statistical independence; per-dongle
CIs bootstrap over routes. Cross-dongle distributions operate on per-dongle
pooled estimates only -- never on raw routes -- so one chatty dongle cannot
dominate a fleet plot.

The hierarchical pooling option (default) computes per-dongle K(v, c) means
first, then takes an UNWEIGHTED mean across dongles for the fleet estimate.
That eliminates dongle-skew at the cost of per-dongle CIs needing to be
wide enough to swamp the unweighted-mean noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from openpilot.tools.lateral_fleet import cache

MIN_BUCKET_COUNT_PER_DONGLE = 100
MIN_ROUTES_PER_DONGLE_CELL = 2
DEFAULT_MIN_DONGLES = 5
DEFAULT_BOOTSTRAP = 200

NUMERIC_STAT_COLUMNS = [
  'resid_yaw_mean', 'resid_yaw_median', 'resid_yaw_iqr',
  'resid_eps_mean', 'resid_eps_median', 'resid_eps_iqr',
  'gain_yaw_mean', 'gain_yaw_median',
  'gain_eps_mean', 'gain_eps_median',
  'torque_driver_mean', 'hca_power_mean',
  'steer_ratio_mean', 'stiffness_factor_mean', 'lateral_delay_mean',
]
KEY_COLUMNS = ['speed_idx', 'curv_idx', 'sign']


def iter_ok_routes(cache_root: Path = cache.CACHE_ROOT) -> Iterator[cache.RouteStatus]:
  if not cache_root.exists():
    return
  for dongle_dir in cache_root.iterdir():
    if not dongle_dir.is_dir():
      continue
    for status_file in dongle_dir.glob('*.status.json'):
      route_id = status_file.name[: -len('.status.json')]
      rs = cache.read_status(dongle_dir.name, route_id)
      if rs is not None and rs.status == 'ok':
        yield rs


def load_dongle_buckets(dongle_id: str, route_ids: list[str]) -> pd.DataFrame:
  frames = []
  for r in route_ids:
    try:
      frames.append(cache.read_buckets(dongle_id, r))
    except (FileNotFoundError, OSError):
      continue
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
  values = np.asarray(values, dtype=np.float64)
  weights = np.asarray(weights, dtype=np.float64)
  ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
  if not ok.any():
    return float('nan')
  return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def _bootstrap_ci(values: np.ndarray, weights: np.ndarray,
                  n: int = DEFAULT_BOOTSTRAP, seed: int = 1) -> tuple[float, float]:
  ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
  v = values[ok]
  w = weights[ok]
  if v.size < MIN_ROUTES_PER_DONGLE_CELL:
    return float('nan'), float('nan')
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, v.size, size=(n, v.size))
  vb = v[idx]
  wb = w[idx]
  means = np.sum(vb * wb, axis=1) / np.sum(wb, axis=1)
  return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def pool_dongle(buckets: pd.DataFrame, n_bootstrap: int = DEFAULT_BOOTSTRAP) -> pd.DataFrame:
  if buckets.empty:
    return pd.DataFrame()
  rows: list[dict] = []
  for (si, ci, sg), grp in buckets.groupby(KEY_COLUMNS, sort=False):
    total = int(grp['count'].sum())
    n_routes = int(grp.shape[0])
    sufficient = (total >= MIN_BUCKET_COUNT_PER_DONGLE) and (n_routes >= MIN_ROUTES_PER_DONGLE_CELL)
    row: dict = {
      'speed_idx': int(si), 'curv_idx': int(ci), 'sign': int(sg),
      'count': total, 'n_routes': n_routes, 'sufficient': sufficient,
    }
    weights = grp['count'].to_numpy().astype(np.float64)
    for col in NUMERIC_STAT_COLUMNS:
      vals = grp[col].to_numpy().astype(np.float64)
      row[col] = _weighted_mean(vals, weights)
      if sufficient and n_bootstrap > 0:
        lo, hi = _bootstrap_ci(vals, weights, n=n_bootstrap,
                               seed=hash((si, ci, sg, col)) & 0xFFFFFFFF)
      else:
        lo, hi = float('nan'), float('nan')
      row[f'{col}_lo'] = lo
      row[f'{col}_hi'] = hi
    rows.append(row)
  return pd.DataFrame(rows)


def per_dongle_metadata(buckets: pd.DataFrame) -> dict:
  if buckets.empty:
    return {}
  w = buckets['count'].to_numpy().astype(np.float64)
  out: dict = {}
  for col in ['steer_ratio_mean', 'stiffness_factor_mean', 'lateral_delay_mean',
              'torque_driver_mean', 'hca_power_mean']:
    out[col] = _weighted_mean(buckets[col].to_numpy(), w)
  out['fingerprint'] = buckets['fingerprint'].iloc[0] if 'fingerprint' in buckets.columns else ''
  out['vin'] = buckets['vin'].iloc[0] if 'vin' in buckets.columns else ''
  out['lcp_seen'] = bool(buckets['lcp_seen'].any()) if 'lcp_seen' in buckets.columns else False
  out['total_routes'] = int(buckets[['dongle_id', 'route_id']].drop_duplicates().shape[0]) \
    if {'dongle_id', 'route_id'}.issubset(buckets.columns) else 0
  out['total_samples'] = int(buckets['count'].sum())
  return out


def cross_dongle(pooled_by_dongle: dict[str, pd.DataFrame],
                 min_dongles: int = DEFAULT_MIN_DONGLES,
                 hierarchical: bool = True) -> pd.DataFrame:
  """Cross-dongle pooled estimates per bucket.

  hierarchical=True: per-dongle pooled values (already in pooled_by_dongle)
  are aggregated by UNWEIGHTED mean across dongles. This is what the user
  requested as default for this fleet, to eliminate the 1333/1662 dongle
  skew. CIs are bootstrap over DONGLES.
  """
  long_frames = []
  for dongle, pooled in pooled_by_dongle.items():
    if pooled.empty:
      continue
    p = pooled.copy()
    p['dongle_id'] = dongle
    long_frames.append(p)
  if not long_frames:
    return pd.DataFrame()
  long_df = pd.concat(long_frames, ignore_index=True)
  long_df = long_df[long_df['sufficient']]

  rows = []
  for (si, ci, sg), grp in long_df.groupby(KEY_COLUMNS, sort=False):
    n_d = int(grp.shape[0])
    row = {'speed_idx': int(si), 'curv_idx': int(ci), 'sign': int(sg),
           'n_dongles': n_d, 'insufficient_dongles': n_d < min_dongles}
    for col in NUMERIC_STAT_COLUMNS:
      vals = grp[col].to_numpy().astype(np.float64)
      vals = vals[np.isfinite(vals)]
      if vals.size == 0 or n_d < min_dongles:
        row[f'{col}_p50'] = float('nan')
        row[f'{col}_p05'] = float('nan')
        row[f'{col}_p95'] = float('nan')
        row[f'{col}_mean'] = float('nan')
        row[f'{col}_boot_lo'] = float('nan')
        row[f'{col}_boot_hi'] = float('nan')
      else:
        row[f'{col}_p50'] = float(np.percentile(vals, 50))
        row[f'{col}_p05'] = float(np.percentile(vals, 5))
        row[f'{col}_p95'] = float(np.percentile(vals, 95))
        if hierarchical:
          # Unweighted mean across dongles plus bootstrap CI over dongles.
          row[f'{col}_mean'] = float(np.mean(vals))
          rng = np.random.default_rng(hash((si, ci, sg, col)) & 0xFFFFFFFF)
          idx = rng.integers(0, vals.size, size=(DEFAULT_BOOTSTRAP, vals.size))
          means = np.mean(vals[idx], axis=1)
          row[f'{col}_boot_lo'] = float(np.percentile(means, 5))
          row[f'{col}_boot_hi'] = float(np.percentile(means, 95))
        else:
          row[f'{col}_mean'] = float(np.mean(vals))
          row[f'{col}_boot_lo'] = row[f'{col}_p05']
          row[f'{col}_boot_hi'] = row[f'{col}_p95']
    rows.append(row)
  return pd.DataFrame(rows)


def aggregate_run(run_dir: Path,
                  cache_root: Path = cache.CACHE_ROOT,
                  n_bootstrap: int = DEFAULT_BOOTSTRAP,
                  min_dongles: int = DEFAULT_MIN_DONGLES,
                  hierarchical: bool = True) -> dict:
  run_dir = Path(run_dir)
  (run_dir / 'dongle_buckets').mkdir(parents=True, exist_ok=True)
  (run_dir / 'dongle_pooled').mkdir(parents=True, exist_ok=True)

  routes_rows: list[dict] = []
  by_dongle_routes: dict[str, list[str]] = {}
  for rs in iter_ok_routes(cache_root):
    routes_rows.append({
      'dongle_id': rs.dongle_id, 'route_id': rs.route_id,
      'fingerprint': rs.fingerprint, 'vin': rs.vin, 'can_bus': rs.can_bus,
      'duration_engaged_s': rs.duration_engaged_s,
      'duration_strict_gated_s': rs.duration_strict_gated_s,
      'lcp_seen': 'lcp_seen=True' in (rs.message or ''),
    })
    by_dongle_routes.setdefault(rs.dongle_id, []).append(rs.route_id)

  pd.DataFrame(routes_rows).to_parquet(run_dir / 'routes.parquet', index=False)

  pooled_by_dongle: dict[str, pd.DataFrame] = {}
  dongle_meta_rows: list[dict] = []
  for dongle, route_ids in by_dongle_routes.items():
    buckets = load_dongle_buckets(dongle, route_ids)
    if buckets.empty:
      continue
    buckets.to_parquet(run_dir / 'dongle_buckets' / f'{dongle}.parquet', index=False)
    pooled = pool_dongle(buckets, n_bootstrap=n_bootstrap)
    pooled.to_parquet(run_dir / 'dongle_pooled' / f'{dongle}.parquet', index=False)
    pooled_by_dongle[dongle] = pooled
    meta = per_dongle_metadata(buckets)
    meta['dongle_id'] = dongle
    dongle_meta_rows.append(meta)

  pd.DataFrame(dongle_meta_rows).to_parquet(run_dir / 'dongles.parquet', index=False)
  cross = cross_dongle(pooled_by_dongle, min_dongles=min_dongles, hierarchical=hierarchical)
  cross.to_parquet(run_dir / 'cross_dongle.parquet', index=False)

  return {
    'n_routes': len(routes_rows),
    'n_dongles': len(by_dongle_routes),
    'n_dongles_with_pooled': len(pooled_by_dongle),
    'hierarchical': hierarchical,
    'run_dir': str(run_dir),
  }
