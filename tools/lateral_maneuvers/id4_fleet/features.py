"""Streaming summarizers. No full-route timelines held in RAM.

The route emits one fixed-size feature dict (~50 KB packed). All accumulators
are O(1) memory per pair / per bucket.

Bucketing mirrors CurvatureDLookup so per-route features can be compared
1:1 against what the sunnypilot learner sees.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


SPEED_ANCHORS = (np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]) / 3.6).astype(np.float64)
CURVATURE_BUCKET_EDGES = np.array([
  1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5,
  1.28e-4, 2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3,
], dtype=np.float64)
N_SPEED = len(SPEED_ANCHORS)
N_CURVATURE = len(CURVATURE_BUCKET_EDGES) - 1   # = 12

LAG_GRID_S = np.arange(-0.5, 0.5 + 1e-9, 0.05, dtype=np.float64)   # 21 bins
LAG_BIN_HALF = 0.025
DEADBAND_EDGES = np.array([0.0, 1.0e-5, 1.0e-4], dtype=np.float64)


def speed_index(v_ego: float) -> int | None:
  v = float(v_ego)
  if v < SPEED_ANCHORS[0] * 0.5:
    return None
  return int(np.argmin(np.abs(SPEED_ANCHORS - v)))


def curvature_index(curvature: float) -> int | None:
  c = abs(float(curvature))
  if c < CURVATURE_BUCKET_EDGES[0] or c > CURVATURE_BUCKET_EDGES[-1]:
    return None
  idx = int(np.searchsorted(CURVATURE_BUCKET_EDGES, c, side="right") - 1)
  return int(np.clip(idx, 0, N_CURVATURE - 1))


@dataclass
class BucketAccumulator:
  """Per-pair bucketed residual stats. Residual = desired - actual (no direction folding)."""
  name: str
  count: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  sum_residual: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  sum_abs_residual: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  sumsq_residual: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  # sign-stratified (the asymmetry the learner cannot see, by design)
  count_pos: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  count_neg: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  sum_residual_pos: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  sum_residual_neg: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  saturation_count: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64))
  # deadband (stratified by |desired| in [0, 1e-5, 1e-4] bins)
  deadband_count: np.ndarray = field(default_factory=lambda: np.zeros(len(DEADBAND_EDGES), dtype=np.float64))
  deadband_sum_residual: np.ndarray = field(default_factory=lambda: np.zeros(len(DEADBAND_EDGES), dtype=np.float64))
  deadband_sumsq_residual: np.ndarray = field(default_factory=lambda: np.zeros(len(DEADBAND_EDGES), dtype=np.float64))

  def add(self, desired: float, actual: float, v_ego: float) -> None:
    residual = desired - actual
    self._deadband_add(desired, residual)

    s_idx = speed_index(v_ego)
    c_idx = curvature_index(desired)
    if s_idx is None or c_idx is None:
      return
    self.count[s_idx, c_idx] += 1.0
    self.sum_residual[s_idx, c_idx] += residual
    self.sum_abs_residual[s_idx, c_idx] += abs(residual)
    self.sumsq_residual[s_idx, c_idx] += residual * residual
    if desired >= 0.0:
      self.count_pos[s_idx, c_idx] += 1.0
      self.sum_residual_pos[s_idx, c_idx] += residual
    else:
      self.count_neg[s_idx, c_idx] += 1.0
      self.sum_residual_neg[s_idx, c_idx] += residual
    if abs(residual) > 0.5 * abs(desired):
      self.saturation_count[s_idx, c_idx] += 1.0

  def _deadband_add(self, desired: float, residual: float) -> None:
    abs_d = abs(desired)
    if abs_d >= DEADBAND_EDGES[-1]:
      return
    bin_idx = int(np.searchsorted(DEADBAND_EDGES, abs_d, side="right") - 1)
    bin_idx = int(np.clip(bin_idx, 0, len(DEADBAND_EDGES) - 1))
    self.deadband_count[bin_idx] += 1.0
    self.deadband_sum_residual[bin_idx] += residual
    self.deadband_sumsq_residual[bin_idx] += residual * residual

  def to_dict(self) -> dict:
    return {
      f"{self.name}_count": self.count,
      f"{self.name}_sum_residual": self.sum_residual,
      f"{self.name}_sum_abs_residual": self.sum_abs_residual,
      f"{self.name}_sumsq_residual": self.sumsq_residual,
      f"{self.name}_count_pos": self.count_pos,
      f"{self.name}_count_neg": self.count_neg,
      f"{self.name}_sum_residual_pos": self.sum_residual_pos,
      f"{self.name}_sum_residual_neg": self.sum_residual_neg,
      f"{self.name}_saturation_count": self.saturation_count,
      f"{self.name}_deadband_count": self.deadband_count,
      f"{self.name}_deadband_sum_residual": self.deadband_sum_residual,
      f"{self.name}_deadband_sumsq_residual": self.deadband_sumsq_residual,
    }


@dataclass
class XcorrAccumulator:
  """Lagged Pearson correlation between desired and actual across the LAG_GRID_S grid.
  Streaming: maintains a deque of recent (t, desired, actual) within +/- LAG_GRID_S[-1] seconds."""
  name: str
  history_s: float = float(abs(LAG_GRID_S[0])) + 0.5    # extra slack
  _history: deque = field(default_factory=deque)
  n: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))
  sx: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))
  sy: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))
  sxx: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))
  syy: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))
  sxy: np.ndarray = field(default_factory=lambda: np.zeros(len(LAG_GRID_S), dtype=np.float64))

  def add(self, t: float, desired_now: float, actual_now: float) -> None:
    cutoff = t - self.history_s
    while self._history and self._history[0][0] < cutoff:
      self._history.popleft()

    for t_p, desired_p, actual_p in self._history:
      dt = t - t_p
      # bin (desired_p -> actual_now) at lag = +dt (desired leads actual)
      if LAG_GRID_S[0] - LAG_BIN_HALF <= dt <= LAG_GRID_S[-1] + LAG_BIN_HALF:
        idx = int(np.argmin(np.abs(LAG_GRID_S - dt)))
        self._accum(idx, desired_p, actual_now)
      # bin (desired_now -> actual_p) at lag = -dt (actual leads desired)
      neg = -dt
      if LAG_GRID_S[0] - LAG_BIN_HALF <= neg <= LAG_GRID_S[-1] + LAG_BIN_HALF:
        idx = int(np.argmin(np.abs(LAG_GRID_S - neg)))
        self._accum(idx, desired_now, actual_p)

    # zero-lag contribution
    if abs(0.0 - LAG_GRID_S).min() < LAG_BIN_HALF:
      zero_idx = int(np.argmin(np.abs(LAG_GRID_S)))
      self._accum(zero_idx, desired_now, actual_now)

    self._history.append((t, desired_now, actual_now))

  def _accum(self, idx: int, x: float, y: float) -> None:
    self.n[idx] += 1.0
    self.sx[idx] += x
    self.sy[idx] += y
    self.sxx[idx] += x * x
    self.syy[idx] += y * y
    self.sxy[idx] += x * y

  def correlations(self) -> np.ndarray:
    n = self.n
    valid = n > 1
    num = n * self.sxy - self.sx * self.sy
    denx = n * self.sxx - self.sx * self.sx
    deny = n * self.syy - self.sy * self.sy
    denom = np.sqrt(np.maximum(denx * deny, 0.0))
    out = np.where(valid & (denom > 0), num / np.where(denom > 0, denom, 1.0), np.nan)
    return out

  def to_dict(self) -> dict:
    return {
      f"{self.name}_xcorr_n": self.n.copy(),
      f"{self.name}_xcorr_r": self.correlations(),
      f"{self.name}_xcorr_sxy": self.sxy.copy(),
    }


@dataclass
class ConditioningHistograms:
  """Coarse per-route histograms of conditioning variables. Used as covariates in hypothesis fits."""
  vego_edges: np.ndarray = field(default_factory=lambda: np.arange(0.0, 45.0, 1.0))   # 0..45 m/s, 1 m/s
  vego: np.ndarray = field(default_factory=lambda: np.zeros(45, dtype=np.float64))
  lat_accel_edges: np.ndarray = field(default_factory=lambda: np.arange(0.0, 4.0, 0.1))
  lat_accel: np.ndarray = field(default_factory=lambda: np.zeros(40, dtype=np.float64))
  steer_ratio_sum: float = 0.0
  steer_ratio_sumsq: float = 0.0
  steer_ratio_n: float = 0.0
  stiffness_sum: float = 0.0
  stiffness_sumsq: float = 0.0
  stiffness_n: float = 0.0
  lateral_delay_sum: float = 0.0
  lateral_delay_sumsq: float = 0.0
  lateral_delay_n: float = 0.0
  eps_power_sum: float = 0.0
  eps_power_sumsq: float = 0.0
  eps_power_n: float = 0.0
  driver_torque_sum_abs: float = 0.0
  driver_torque_n: float = 0.0

  def add_sample(self, v_ego: float, lat_accel: float) -> None:
    iv = int(np.clip(v_ego, 0.0, 44.999))
    self.vego[iv] += 1.0
    ia = int(np.clip(abs(lat_accel) / 0.1, 0.0, 39.999))
    self.lat_accel[ia] += 1.0

  def add_steer_ratio(self, sr: float) -> None:
    self.steer_ratio_sum += sr
    self.steer_ratio_sumsq += sr * sr
    self.steer_ratio_n += 1.0

  def add_stiffness(self, sf: float) -> None:
    self.stiffness_sum += sf
    self.stiffness_sumsq += sf * sf
    self.stiffness_n += 1.0

  def add_lateral_delay(self, ld: float) -> None:
    self.lateral_delay_sum += ld
    self.lateral_delay_sumsq += ld * ld
    self.lateral_delay_n += 1.0

  def add_eps_power(self, p: float) -> None:
    self.eps_power_sum += p
    self.eps_power_sumsq += p * p
    self.eps_power_n += 1.0

  def add_driver_torque(self, nm: float) -> None:
    self.driver_torque_sum_abs += abs(nm)
    self.driver_torque_n += 1.0

  @staticmethod
  def _mean_std(s: float, ss: float, n: float) -> tuple[float, float]:
    if n <= 0:
      return float("nan"), float("nan")
    mean = s / n
    var = max(0.0, ss / n - mean * mean)
    return float(mean), float(var ** 0.5)

  def to_dict(self) -> dict:
    sr_mean, sr_std = self._mean_std(self.steer_ratio_sum, self.steer_ratio_sumsq, self.steer_ratio_n)
    sf_mean, sf_std = self._mean_std(self.stiffness_sum, self.stiffness_sumsq, self.stiffness_n)
    ld_mean, ld_std = self._mean_std(self.lateral_delay_sum, self.lateral_delay_sumsq, self.lateral_delay_n)
    ep_mean, ep_std = self._mean_std(self.eps_power_sum, self.eps_power_sumsq, self.eps_power_n)
    dt_mean = self.driver_torque_sum_abs / self.driver_torque_n if self.driver_torque_n > 0 else float("nan")
    return {
      "hist_vego": self.vego,
      "hist_lat_accel": self.lat_accel,
      "steer_ratio_mean": sr_mean, "steer_ratio_std": sr_std,
      "stiffness_mean": sf_mean, "stiffness_std": sf_std,
      "lateral_delay_mean": ld_mean, "lateral_delay_std": ld_std,
      "eps_power_mean": ep_mean, "eps_power_std": ep_std,
      "driver_torque_mean_abs": dt_mean,
    }
