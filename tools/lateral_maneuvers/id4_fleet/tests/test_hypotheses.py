"""Hypothesis fits recover known truth, and refuse unphysical fits."""
import numpy as np

from tools.lateral_maneuvers.id4_fleet.features import (
  BucketAccumulator, XcorrAccumulator, SPEED_ANCHORS, CURVATURE_BUCKET_EDGES,
)
from tools.lateral_maneuvers.id4_fleet.hypotheses import (
  fit_gain, fit_lag, fit_asymmetry, fit_deadband, fit_null,
  rank_hypotheses, score_dongle,
)


CENTERS = np.sqrt(CURVATURE_BUCKET_EDGES[:-1] * CURVATURE_BUCKET_EDGES[1:])


def _populate(K_pos=0.7, K_neg=0.7, noise=1e-7, n_per_bucket=200, deadband=0.0):
  """Returns a features-dict for one pair (P1_desired_yaw) for testing fits."""
  acc = BucketAccumulator("P1_desired_yaw")
  rng = np.random.default_rng(0)
  for s_idx, v in enumerate(SPEED_ANCHORS):
    for c_idx, c_center in enumerate(CENTERS):
      for _ in range(n_per_bucket):
        for sign in (+1.0, -1.0):
          d = sign * float(c_center)
          K = K_pos if sign > 0 else K_neg
          a_mag = K * max(abs(d) - deadband, 0.0)
          a = (1.0 if d > 0 else -1.0) * a_mag + rng.normal(0, noise)
          acc.add(d, a, float(v))
  return acc.to_dict()


def test_fit_gain_recovers_K_07():
  f = _populate(K_pos=0.7, K_neg=0.7)
  fit = fit_gain(f["P1_desired_yaw_count_pos"], f["P1_desired_yaw_count_neg"],
                 f["P1_desired_yaw_sum_residual_pos"], f["P1_desired_yaw_sum_residual_neg"])
  assert not fit.failed, fit.reason
  assert abs(fit.params["K"] - 0.7) < 0.02


def test_fit_gain_rejects_unphysical_K():
  # Pretend the car overshoots: K = 2.0 (impossible for passive plant)
  f = _populate(K_pos=2.0, K_neg=2.0)
  fit = fit_gain(f["P1_desired_yaw_count_pos"], f["P1_desired_yaw_count_neg"],
                 f["P1_desired_yaw_sum_residual_pos"], f["P1_desired_yaw_sum_residual_neg"])
  assert fit.failed
  assert "unphysical" in fit.reason


def test_fit_asymmetry_recovers_K_pos_K_neg():
  f = _populate(K_pos=0.9, K_neg=0.6)
  fit = fit_asymmetry(
    f["P1_desired_yaw_count_pos"], f["P1_desired_yaw_count_neg"],
    f["P1_desired_yaw_sum_residual_pos"], f["P1_desired_yaw_sum_residual_neg"],
  )
  assert not fit.failed, fit.reason
  assert abs(fit.params["K_pos"] - 0.9) < 0.02
  assert abs(fit.params["K_neg"] - 0.6) < 0.02
  assert abs(fit.params["delta"] - 0.3) < 0.04


def test_fit_lag_recovers_200ms():
  dt = 0.04
  ts = np.arange(0, 60.0, dt)
  desired = 1e-4 * np.sin(2 * np.pi * 0.3 * ts)
  lag_steps = int(round(0.20 / dt))
  actual = np.empty_like(desired)
  actual[:lag_steps] = 0.0
  actual[lag_steps:] = desired[:-lag_steps] * 0.8
  acc = XcorrAccumulator("P1")
  for i, t in enumerate(ts):
    acc.add(float(t), float(desired[i]), float(actual[i]))
  r = acc.correlations()
  fit = fit_lag(r, gated_n=len(ts))
  assert not fit.failed, fit.reason
  assert abs(fit.params["tau_s"] - 0.20) < 0.05


def test_fit_deadband_detects_zero_stratum_signature():
  # Manufacture: constant 1e-4 residual at d=+5e-6 (in deadband stratum), tight tracking elsewhere.
  acc = BucketAccumulator("P1_desired_yaw")
  rng = np.random.default_rng(0)
  for _ in range(200):
    d = 5.0e-6
    a = d - 1.0e-4
    acc.add(d, a, 22.0)
  # outside-deadband samples (curvature centers >= 1e-5) with tight tracking
  for s_idx, v in enumerate(SPEED_ANCHORS):
    for c_idx, c in enumerate(CENTERS):
      if c < 1.0e-5:
        continue
      for _ in range(20):
        for sign in (+1.0, -1.0):
          d = sign * float(c)
          a = d * 0.99 + rng.normal(0, 1e-8)     # tiny residual, symmetric
          acc.add(d, a, float(v))
  f = acc.to_dict()
  fit = fit_deadband(
    f["P1_desired_yaw_count"],
    f["P1_desired_yaw_sum_residual_pos"], f["P1_desired_yaw_sum_residual_neg"],
    f["P1_desired_yaw_count_pos"], f["P1_desired_yaw_count_neg"],
    f["P1_desired_yaw_deadband_count"], f["P1_desired_yaw_deadband_sum_residual"],
  )
  assert not fit.failed, fit.reason
  assert fit.params["d_est"] > 5.0e-5


def test_rank_picks_lowest_aic_when_gap_exceeds_2():
  from tools.lateral_maneuvers.id4_fleet.hypotheses import HypothesisFit
  fits = [
    HypothesisFit("H_null", {}, rss=10.0, n=1000, k=0),
    HypothesisFit("H_gain", {}, rss=1.0,  n=1000, k=1),
    HypothesisFit("H_lag",  {}, rss=2.0,  n=1000, k=1),
  ]
  winner, _ = rank_hypotheses(fits)
  assert winner == "H_gain"


def test_rank_returns_mixed_when_ties():
  from tools.lateral_maneuvers.id4_fleet.hypotheses import HypothesisFit
  fits = [
    HypothesisFit("H_null", {}, rss=10.0, n=1000, k=0),
    HypothesisFit("H_gain", {}, rss=1.0,  n=1000, k=1),
    HypothesisFit("H_lag",  {}, rss=1.001, n=1000, k=1),
  ]
  winner, _ = rank_hypotheses(fits)
  assert winner == "mixed"


def test_score_dongle_full_pipeline_with_known_gain():
  f = _populate(K_pos=0.65, K_neg=0.65)
  # Add a dummy xcorr_r so fit_lag has something to work with (no real lag here)
  f["P1_desired_yaw_xcorr_r"] = np.full(21, 0.3)         # weak, should be flagged failed
  out = score_dongle(f)
  assert out["winner"] == "H_gain"
  assert not out["fits"]["H_gain"]["failed"]
  assert out["fits"]["H_lag"]["failed"]
