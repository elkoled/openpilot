"""Streaming summarizer correctness.

Synthesizes (desired, actual) pairs with known per-bucket residual mean and
known cross-correlation peak; pipes them through the streaming accumulator
and asserts the recovered values match a slow reference within tight tolerances.
"""
import numpy as np

from tools.lateral_maneuvers.id4_fleet.features import (
  BucketAccumulator, ConditioningHistograms, XcorrAccumulator,
  SPEED_ANCHORS, CURVATURE_BUCKET_EDGES, LAG_GRID_S,
  curvature_index, speed_index,
)


def test_bucket_accumulator_per_bucket_mean_and_sumsq():
  rng = np.random.default_rng(42)
  v = float(SPEED_ANCHORS[3])           # 80 kph
  c_center = float(np.sqrt(CURVATURE_BUCKET_EDGES[6] * CURVATURE_BUCKET_EDGES[7]))
  s_idx = speed_index(v); c_idx = curvature_index(c_center)
  assert s_idx == 3 and c_idx == 6

  acc = BucketAccumulator("test")
  K = 0.7
  noise = 1e-7
  desireds = []
  actuals = []
  for _ in range(2000):
    d = c_center * float(rng.uniform(0.95, 1.05))
    a = K * d + rng.normal(0, noise)
    acc.add(d, a, v)
    desireds.append(d); actuals.append(a)

  residuals = np.array(desireds) - np.array(actuals)
  expected_mean = float(residuals.mean())
  expected_sumsq = float((residuals ** 2).sum())

  assert acc.count[s_idx, c_idx] == 2000
  got_mean = float(acc.sum_residual[s_idx, c_idx] / acc.count[s_idx, c_idx])
  assert abs(got_mean - expected_mean) < 1e-12
  assert abs(acc.sumsq_residual[s_idx, c_idx] - expected_sumsq) < 1e-12


def test_bucket_accumulator_sign_stratification():
  v = float(SPEED_ANCHORS[3])
  c_center = float(np.sqrt(CURVATURE_BUCKET_EDGES[6] * CURVATURE_BUCKET_EDGES[7]))
  s_idx = speed_index(v); c_idx = curvature_index(c_center)
  acc = BucketAccumulator("test")
  for _ in range(100):
    acc.add(+c_center, +c_center * 0.7, v)    # right turn, 70% gain
    acc.add(-c_center, -c_center * 0.5, v)    # left turn, 50% gain (asymmetric)
  assert acc.count_pos[s_idx, c_idx] == 100
  assert acc.count_neg[s_idx, c_idx] == 100
  mean_pos = acc.sum_residual_pos[s_idx, c_idx] / acc.count_pos[s_idx, c_idx]
  mean_neg = acc.sum_residual_neg[s_idx, c_idx] / acc.count_neg[s_idx, c_idx]
  # positive desired (+c) actual=0.7c -> residual=+0.3c
  # negative desired (-c) actual=-0.5c -> residual=-0.5c
  assert abs(mean_pos - 0.30 * c_center) < 1e-15
  assert abs(mean_neg - (-0.50 * c_center)) < 1e-15


def test_xcorr_recovers_known_lag():
  rng = np.random.default_rng(0)
  dt = 0.04
  ts = np.arange(0, 60.0, dt)
  desired = 1e-4 * np.sin(2 * np.pi * 0.3 * ts)
  true_lag_s = 0.20
  lag_steps = int(round(true_lag_s / dt))
  actual = np.empty_like(desired)
  actual[:lag_steps] = 0.0
  actual[lag_steps:] = desired[:-lag_steps] * 0.8

  acc = XcorrAccumulator("test")
  for i, t in enumerate(ts):
    acc.add(float(t), float(desired[i]), float(actual[i]))

  r = acc.correlations()
  best_idx = int(np.nanargmax(r))
  recovered_lag = float(LAG_GRID_S[best_idx])
  assert abs(recovered_lag - true_lag_s) < 0.05
  assert r[best_idx] > 0.95


def test_deadband_features():
  v = float(SPEED_ANCHORS[3])
  acc = BucketAccumulator("test")
  # Add 100 samples near zero with consistent residual = +5e-5 (deadband signature: car undershoots)
  for _ in range(100):
    acc.add(2.0e-6, 2.0e-6 - 5.0e-5, v)         # falls in deadband_count[0] (|d| in [0,1e-5))
  # Add 100 samples in deadband_count[1] with small residual
  for _ in range(100):
    acc.add(5.0e-5, 5.0e-5 - 1.0e-6, v)
  assert acc.deadband_count[0] == 100
  assert acc.deadband_count[1] == 100
  mean0 = acc.deadband_sum_residual[0] / acc.deadband_count[0]
  mean1 = acc.deadband_sum_residual[1] / acc.deadband_count[1]
  assert abs(mean0 - 5.0e-5) < 1e-15
  assert abs(mean1 - 1.0e-6) < 1e-15


def test_conditioning_histograms_running_stats():
  cond = ConditioningHistograms()
  for sr in [14.0, 14.1, 14.2, 13.9, 14.05]:
    cond.add_steer_ratio(sr)
  d = cond.to_dict()
  assert abs(d["steer_ratio_mean"] - 14.05) < 1e-12
  # std of [14.0, 14.1, 14.2, 13.9, 14.05] using population formula
  expected_std = float(np.std([14.0, 14.1, 14.2, 13.9, 14.05]))
  assert abs(d["steer_ratio_std"] - expected_std) < 1e-12
