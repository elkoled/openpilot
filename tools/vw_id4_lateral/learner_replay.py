#!/usr/bin/env python3
"""
Stage 3 — Vendored CurvatureD learner + offline batch-fit comparator.

CurvatureDLookup classmethods and the per-bucket bias-EMA update logic are
vendored verbatim from sunnypilot's openpilot10 fork. The vendored math is
classmethod-only and has no openpilot/cereal/Params dependency, so it can run
offline against pre-extracted 100 Hz arrays.

Source provenance:
  repo: /home/batman/openpilot10
  branch: virtual/dynamic_steering
  HEAD sha at vendor time: febd9128d8e03b551b47d1a69b778aae6edf1136
  curvatured.py last-touch sha: ba93c4129cd5bfb1a9b28d59a7b7879d6fec7c3a

This file is self-contained: it depends only on numpy. Two public entry points:

  - OfflineCurvatureEstimator: replays the bucket-EMA update against a
    pre-gated sequence of (desired, actual, v_ego). Returns the converged
    bias and counts.

  - batch_fit_per_bucket: weighted least-squares per-bucket fit on the same
    samples. Weights = 1 / max(yaw_std, eps)**2. The "ideal" answer to
    compare the learner's online convergence against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ----- vendored from openpilot10/selfdrive/locationd/curvatured.py -----

class CurvatureDLookup:
  SPEED_ANCHORS = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=np.float32) / 3.6
  CURVATURE_BUCKET_EDGES = np.array([
    1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5,
    1.28e-4, 2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3,
  ], dtype=np.float32)
  CURVATURE_BUCKET_CENTERS = np.sqrt(CURVATURE_BUCKET_EDGES[:-1] * CURVATURE_BUCKET_EDGES[1:]).astype(np.float32)
  CURVATURE_BUCKET_MIN = float(CURVATURE_BUCKET_EDGES[0])
  CURVATURE_MIN = 0.0
  CURVATURE_BUCKET_MAX = float(CURVATURE_BUCKET_EDGES[-1])
  LAST_BUCKET_WIDTH = float(CURVATURE_BUCKET_EDGES[-1] - CURVATURE_BUCKET_EDGES[-2])
  CURVATURE_MAX = CURVATURE_BUCKET_MAX + LAST_BUCKET_WIDTH

  MIN_SPEED = float(SPEED_ANCHORS[0] * 0.5)
  MAX_LAT_ACCEL_APPLY = 1.0
  RELATIVE_CAP_FULL_RATIO = 0.50

  MAX_SAMPLES = 600
  MEAN_WINDOW = 180.0
  MIN_REQUIRED_SUPPORT_BUCKETS = 4
  SUPPORT_REFERENCE_LAT_ACCEL = 0.05
  MIN_BUCKET_POINTS = np.array([20, 20, 18, 16, 14, 12, 10, 8, 6, 6, 4, 4], dtype=np.float32)
  FULL_BUCKET_STRENGTH_SAMPLES = MIN_BUCKET_POINTS + MIN_BUCKET_POINTS

  @classmethod
  def bucket_shape(cls):
    return len(cls.SPEED_ANCHORS), len(cls.CURVATURE_BUCKET_CENTERS)

  @classmethod
  def curvature_index(cls, curvature):
    abs_curvature = abs(float(curvature))
    if abs_curvature < cls.CURVATURE_BUCKET_MIN or abs_curvature > cls.CURVATURE_BUCKET_MAX:
      return None
    idx = int(np.searchsorted(cls.CURVATURE_BUCKET_EDGES, abs_curvature, side="right") - 1)
    return min(max(idx, 0), len(cls.CURVATURE_BUCKET_CENTERS) - 1)

  @classmethod
  def speed_interp(cls, v_ego):
    v = float(v_ego)
    if v <= cls.SPEED_ANCHORS[0]:
      return 0, 0, 0.0
    if v >= cls.SPEED_ANCHORS[-1]:
      last = len(cls.SPEED_ANCHORS) - 1
      return last, last, 0.0
    high = int(np.searchsorted(cls.SPEED_ANCHORS, v, side="right"))
    low = high - 1
    span = float(cls.SPEED_ANCHORS[high] - cls.SPEED_ANCHORS[low])
    alpha = (v - float(cls.SPEED_ANCHORS[low])) / max(span, 1e-6)
    return low, high, float(np.clip(alpha, 0.0, 1.0))

  @classmethod
  def learning_speed_weights(cls, v_ego):
    v = float(v_ego)
    if v < cls.MIN_SPEED:
      return []
    low, high, alpha = cls.speed_interp(v)
    if low == high:
      return [(low, 1.0)]
    return [(low, 1.0 - alpha), (high, alpha)]

  @classmethod
  def learning_error_cap(cls, curvature):
    return float(cls.RELATIVE_CAP_FULL_RATIO * abs(float(curvature)))

  @classmethod
  def projected_error(cls, desired_curvature, actual_curvature):
    direction = 1.0 if desired_curvature >= 0.0 else -1.0
    return float(direction * (desired_curvature - actual_curvature))

  @classmethod
  def actual_curvature_from_yaw_rate(cls, yaw_rate, v_ego, roll_compensation=0.0):
    return float(yaw_rate / max(float(v_ego), 0.1) - float(roll_compensation))


# ----- offline replay wrapper -----

@dataclass
class OfflineReplayResult:
  bias: np.ndarray
  counts: np.ndarray
  calibration_percent: float
  num_samples_applied: int
  num_samples_rejected: int


class OfflineCurvatureEstimator(CurvatureDLookup):
  """
  Replays the openpilot10 per-bucket bias EMA against an offline sample stream.

  Each call to step() applies one measurement frame using the exact same math
  as the production daemon's add_measurement(): direction-symmetric error,
  ±50% relative learning cap, two-anchor speed interpolation, EMA with
  delta / min(count, MEAN_WINDOW) horizon, MAX_SAMPLES saturation.
  """

  def __init__(self):
    self.bias = np.zeros(self.bucket_shape(), dtype=np.float64)
    self.counts = np.zeros(self.bucket_shape(), dtype=np.float64)
    self.num_applied = 0
    self.num_rejected = 0

  def step(self, desired_curvature: float, actual_curvature: float, v_ego: float) -> bool:
    curvature_idx = self.curvature_index(desired_curvature)
    speed_weights = self.learning_speed_weights(v_ego)
    if curvature_idx is None or len(speed_weights) == 0:
      self.num_rejected += 1
      return False

    error_cap = self.learning_error_cap(desired_curvature)
    error = float(np.clip(self.projected_error(desired_curvature, actual_curvature),
                          -error_cap, error_cap))

    applied = False
    for speed_idx, weight in speed_weights:
      if weight <= 0.0:
        continue
      prev_count = float(self.counts[speed_idx, curvature_idx])
      sample_count = min(prev_count + float(weight), self.MAX_SAMPLES)
      delta = sample_count - prev_count
      if delta <= 0.0:
        continue
      self.counts[speed_idx, curvature_idx] = sample_count
      alpha = delta / min(sample_count, self.MEAN_WINDOW)
      prev_bias = float(self.bias[speed_idx, curvature_idx])
      self.bias[speed_idx, curvature_idx] = prev_bias + alpha * (error - prev_bias)
      applied = True
    if applied:
      self.num_applied += 1
    else:
      self.num_rejected += 1
    return applied

  def replay(self, desired: np.ndarray, actual: np.ndarray, v_ego: np.ndarray) -> OfflineReplayResult:
    n = len(desired)
    assert len(actual) == n and len(v_ego) == n
    for i in range(n):
      self.step(float(desired[i]), float(actual[i]), float(v_ego[i]))
    cal_pct = self._calibration_percent()
    return OfflineReplayResult(
      bias=self.bias.copy(),
      counts=self.counts.copy(),
      calibration_percent=cal_pct,
      num_samples_applied=self.num_applied,
      num_samples_rejected=self.num_rejected,
    )

  def _calibration_percent(self) -> float:
    """Approximation of the daemon's calibration_percent: per-speed fraction
    of speed-rows whose top-N support buckets are at full strength."""
    fully = 0
    for s in range(len(self.SPEED_ANCHORS)):
      v_ego = float(self.SPEED_ANCHORS[s])
      max_bucket = self.MAX_LAT_ACCEL_APPLY / max(v_ego ** 2, 1e-6)
      max_bucket_idx = self.curvature_index(min(max_bucket, self.CURVATURE_BUCKET_MAX))
      if max_bucket_idx is None:
        continue
      supported = self.counts[s, :max_bucket_idx + 1]
      full = np.sum(supported >= self.FULL_BUCKET_STRENGTH_SAMPLES[:max_bucket_idx + 1])
      typical = self.SUPPORT_REFERENCE_LAT_ACCEL / max(v_ego ** 2, 1e-6)
      need = int(np.clip(int(np.searchsorted(self.CURVATURE_BUCKET_CENTERS, typical, side="right")),
                         self.MIN_REQUIRED_SUPPORT_BUCKETS, len(self.CURVATURE_BUCKET_CENTERS)))
      if full >= need:
        fully += 1
    return 100.0 * fully / float(len(self.SPEED_ANCHORS))


# ----- offline batch fit -----

@dataclass
class BatchFitResult:
  """Per-bucket weighted-least-squares fit on the same gated samples the
  learner would see. This is the 'ideal' bucket bias the learner is trying
  to converge to. Comparing the learner's final bias to this answers
  'does the learner converge to the right answer'."""
  bias: np.ndarray             # shape (7, 12) — weighted-mean signed error per bucket
  bias_var: np.ndarray         # variance of that mean
  counts: np.ndarray           # sample count per bucket
  bias_clipped_fraction: np.ndarray  # fraction of samples that hit the ±50% cap
  shape: tuple = field(default_factory=lambda: CurvatureDLookup.bucket_shape())


def batch_fit_per_bucket(
  desired: np.ndarray,
  actual: np.ndarray,
  v_ego: np.ndarray,
  yaw_std: np.ndarray | None = None,
) -> BatchFitResult:
  """
  Identical gating math to OfflineCurvatureEstimator (same bucket grid, same
  ±50% error cap, same two-anchor speed interpolation), but computed in a
  single pooled pass as weighted means rather than online EMA.

  weight_per_sample := w * speed_weight, where w := 1/max(yaw_std, 1e-3)**2
  if yaw_std provided, else 1.0.
  """
  K = CurvatureDLookup
  S, C = K.bucket_shape()
  num = np.zeros((S, C), dtype=np.float64)
  den = np.zeros((S, C), dtype=np.float64)
  sumsq = np.zeros((S, C), dtype=np.float64)
  counts = np.zeros((S, C), dtype=np.float64)
  clipped = np.zeros((S, C), dtype=np.float64)
  clipped_total = np.zeros((S, C), dtype=np.float64)

  if yaw_std is None:
    w_arr = np.ones_like(desired, dtype=np.float64)
  else:
    w_arr = 1.0 / np.maximum(np.asarray(yaw_std, dtype=np.float64), 1e-3) ** 2

  for i in range(len(desired)):
    d = float(desired[i])
    a = float(actual[i])
    v = float(v_ego[i])
    c_idx = K.curvature_index(d)
    sw = K.learning_speed_weights(v)
    if c_idx is None or len(sw) == 0:
      continue
    cap = K.learning_error_cap(d)
    raw_err = K.projected_error(d, a)
    err = float(np.clip(raw_err, -cap, cap))
    was_clipped = abs(raw_err) > cap
    sw_total = sum(weight for _, weight in sw)
    if sw_total <= 0:
      continue
    for s_idx, sweight in sw:
      ww = float(w_arr[i]) * float(sweight)
      num[s_idx, c_idx] += ww * err
      den[s_idx, c_idx] += ww
      sumsq[s_idx, c_idx] += ww * err * err
      counts[s_idx, c_idx] += float(sweight)
      clipped_total[s_idx, c_idx] += float(sweight)
      if was_clipped:
        clipped[s_idx, c_idx] += float(sweight)

  with np.errstate(invalid="ignore", divide="ignore"):
    bias = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    # weighted variance of the mean: E[err²] - mean² , divided by effective N
    mean_sq = np.where(den > 0, sumsq / np.maximum(den, 1e-12), 0.0)
    eff_n = np.maximum(counts, 1.0)
    bias_var = np.maximum(mean_sq - bias * bias, 0.0) / eff_n
    cf = np.where(clipped_total > 0, clipped / np.maximum(clipped_total, 1e-12), 0.0)

  return BatchFitResult(
    bias=bias.astype(np.float32),
    bias_var=bias_var.astype(np.float32),
    counts=counts.astype(np.float32),
    bias_clipped_fraction=cf.astype(np.float32),
  )
