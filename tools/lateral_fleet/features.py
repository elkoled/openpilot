"""Bucket grid, gate predicates, residual definitions.

Pure math, no I/O. Constants mirror sunnypilot's CurvatureEstimator so that
fleet results are directly comparable to what that learner would have seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRAVITY = 9.81

SPEED_ANCHORS_KPH = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
SPEED_ANCHORS = SPEED_ANCHORS_KPH / 3.6
V_EGO_FLOOR = float(SPEED_ANCHORS[0] * 0.5)

CURVATURE_BUCKET_EDGES = np.array([
  1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5, 1.28e-4,
  2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3,
])
NUM_CURV_BUCKETS = len(CURVATURE_BUCKET_EDGES) - 1
NUM_SPEED_ANCHORS = len(SPEED_ANCHORS)

MAX_LAT_ACCEL_LEARN = 1.0
MAX_LEARN_ROLL_LAT_ACCEL = 0.10        # sunnypilot's strict roll gate (~0.58°)
MAX_FIT_ROLL_LAT_ACCEL = 0.85          # ours: ~5°. Roll does not bias yaw-derived
                                       # curvature (c_yaw = yaw_rate/v); the strict
                                       # gate was for sunnypilot's lat-accel path.
                                       # We keep both so the strict gate is still
                                       # available for direct sunnypilot replay.
MAX_YAW_RATE_STD = 1.0
MIN_ENGAGE_BUFFER_S = 2.0
GAIN_FLOOR_CMD = 5e-5

RESAMPLE_HZ = 50.0
RESAMPLE_DT = 1.0 / RESAMPLE_HZ

CALIB_STATUS_CALIBRATED = 1


def curvature_from_yaw(yaw_rate: np.ndarray, v_ego: np.ndarray, roll_rad: np.ndarray | None = None) -> np.ndarray:
  """Yaw-rate-derived curvature.

  Empirically verified on f73c01590368ee5b/0000000e--2d623b6df3:
  livePose.angularVelocityDevice.z has the same sign as openpilot's
  curvature convention on VW MEB. Roll is unused for yaw-derived curvature.
  """
  v_safe = np.maximum(v_ego, 0.1)
  return yaw_rate / v_safe


def shift_by_delay(arr: np.ndarray, delay_s: float, dt: float = RESAMPLE_DT) -> np.ndarray:
  """Delay `arr` forward in time by `delay_s` (sample-and-hold)."""
  n = int(round(delay_s / dt))
  if n <= 0:
    return arr
  out = np.empty_like(arr)
  out[:n] = arr[0]
  out[n:] = arr[:-n]
  return out


def lat_accel_gate(curvature: np.ndarray, v_ego: np.ndarray) -> np.ndarray:
  return np.abs(curvature) * v_ego ** 2 <= MAX_LAT_ACCEL_LEARN


def roll_gate(roll_rad: np.ndarray, threshold: float = MAX_FIT_ROLL_LAT_ACCEL) -> np.ndarray:
  """Default threshold is `MAX_FIT_ROLL_LAT_ACCEL` (≈ 5° roll). Pass
  `threshold=MAX_LEARN_ROLL_LAT_ACCEL` for sunnypilot-strict semantics.
  """
  return np.abs(GRAVITY * np.sin(roll_rad)) <= threshold


def yaw_rate_std_gate(yaw_rate_std: np.ndarray) -> np.ndarray:
  return yaw_rate_std < MAX_YAW_RATE_STD


def v_ego_floor_gate(v_ego: np.ndarray) -> np.ndarray:
  return v_ego > V_EGO_FLOOR


def engagement_buffer_mask(lat_active: np.ndarray, steering_pressed: np.ndarray,
                           dt: float = RESAMPLE_DT, buffer_s: float = MIN_ENGAGE_BUFFER_S) -> np.ndarray:
  """True where buffer_s have STRICTLY elapsed since the most recent reset
  event (lat-not-active, steering pressed, or rising edge of either).
  """
  n = lat_active.shape[0]
  buffer_n = int(round(buffer_s / dt))
  out = np.zeros(n, dtype=bool)
  samples_since_event = 0
  prev_lat = False
  prev_press = False
  for i in range(n):
    la = bool(lat_active[i])
    sp = bool(steering_pressed[i])
    if (not la) or sp or (la and not prev_lat) or (sp and not prev_press):
      samples_since_event = 0
    else:
      samples_since_event += 1
    out[i] = la and (not sp) and (samples_since_event > buffer_n)
    prev_lat = la
    prev_press = sp
  return out


def strict_gates(v_ego, yaw_rate, yaw_rate_std, roll_rad, lat_active,
                 steering_pressed, calib_status, c_cmd):
  return (
    lat_accel_gate(c_cmd, v_ego)
    & roll_gate(roll_rad)
    & yaw_rate_std_gate(yaw_rate_std)
    & v_ego_floor_gate(v_ego)
    & engagement_buffer_mask(lat_active, steering_pressed)
    & (calib_status == CALIB_STATUS_CALIBRATED)
  )


def loose_gates(lat_active, steering_pressed, v_ego):
  return lat_active & (~steering_pressed) & v_ego_floor_gate(v_ego)


def speed_bucket(v_ego: np.ndarray) -> np.ndarray:
  idx = np.argmin(np.abs(v_ego[:, None] - SPEED_ANCHORS[None, :]), axis=1)
  return idx.astype(np.int32)


def curvature_bucket(curvature: np.ndarray) -> np.ndarray:
  abs_c = np.abs(curvature)
  idx = np.searchsorted(CURVATURE_BUCKET_EDGES, abs_c, side='right') - 1
  in_range = (idx >= 0) & (idx < NUM_CURV_BUCKETS)
  return np.where(in_range, idx, -1).astype(np.int32)


def sign_index(curvature: np.ndarray) -> np.ndarray:
  return np.sign(curvature).astype(np.int8)


STAT_COLUMNS = [
  'speed_idx', 'curv_idx', 'sign', 'count',
  'resid_yaw_mean', 'resid_yaw_median', 'resid_yaw_iqr',
  'resid_eps_mean', 'resid_eps_median', 'resid_eps_iqr',
  'gain_yaw_mean', 'gain_yaw_median',
  'gain_eps_mean', 'gain_eps_median',
  'torque_driver_mean', 'hca_power_mean',
  'steer_ratio_mean', 'stiffness_factor_mean', 'lateral_delay_mean',
]


def _iqr(x: np.ndarray) -> float:
  if x.size == 0:
    return float('nan')
  q75, q25 = np.percentile(x, [75, 25])
  return float(q75 - q25)


def bucket_route(timeline: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
  if mask.sum() == 0:
    return pd.DataFrame(columns=STAT_COLUMNS)
  df = timeline.loc[mask].copy()
  s_idx = speed_bucket(df['v_ego'].to_numpy())
  c_idx = curvature_bucket(df['c_cmd'].to_numpy())
  sign = sign_index(df['c_cmd'].to_numpy())
  df = df.assign(speed_idx=s_idx, curv_idx=c_idx, sign=sign)
  df = df[df['curv_idx'] >= 0]
  if df.empty:
    return pd.DataFrame(columns=STAT_COLUMNS)

  rows = []
  for (si, ci, sg), grp in df.groupby(['speed_idx', 'curv_idx', 'sign'], sort=False):
    r_yaw = grp['resid_yaw'].to_numpy()
    r_eps = grp['resid_eps'].to_numpy()
    g_yaw = grp['gain_yaw'].to_numpy()
    g_eps = grp['gain_eps'].to_numpy()
    rows.append({
      'speed_idx': int(si), 'curv_idx': int(ci), 'sign': int(sg),
      'count': int(grp.shape[0]),
      'resid_yaw_mean': float(np.mean(r_yaw)),
      'resid_yaw_median': float(np.median(r_yaw)),
      'resid_yaw_iqr': _iqr(r_yaw),
      'resid_eps_mean': float(np.mean(r_eps)),
      'resid_eps_median': float(np.median(r_eps)),
      'resid_eps_iqr': _iqr(r_eps),
      'gain_yaw_mean': float(np.nanmean(g_yaw)) if g_yaw.size else float('nan'),
      'gain_yaw_median': float(np.nanmedian(g_yaw)) if g_yaw.size else float('nan'),
      'gain_eps_mean': float(np.nanmean(g_eps)) if g_eps.size else float('nan'),
      'gain_eps_median': float(np.nanmedian(g_eps)) if g_eps.size else float('nan'),
      'torque_driver_mean': float(np.mean(grp['torque_driver'])),
      'hca_power_mean': float(np.mean(grp['hca_power'])),
      'steer_ratio_mean': float(np.mean(grp['steer_ratio'])),
      'stiffness_factor_mean': float(np.mean(grp['stiffness_factor'])),
      'lateral_delay_mean': float(np.mean(grp['lateral_delay'])),
    })
  return pd.DataFrame(rows, columns=STAT_COLUMNS)


def compute_gain_ratio(c_actual: np.ndarray, c_cmd: np.ndarray) -> np.ndarray:
  out = np.full_like(c_cmd, np.nan, dtype=np.float64)
  ok = np.abs(c_cmd) > GAIN_FLOOR_CMD
  out[ok] = c_actual[ok] / c_cmd[ok]
  return out
