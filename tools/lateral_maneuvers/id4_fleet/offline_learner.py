"""Replay a route through the sunnypilot dynamic_steering CurvatureEstimator.

Imports ~/openpilot10's `selfdrive/locationd/curvatured.py` via sys.path append
rather than copying the 928-line file. Used to compare:

  (1) what the learner converges to on a route, vs
  (2) what a clean offline batch fit of the same data produces.

Where the two disagree, the learner is doing one of:
  - hitting its +/-50% bias cap on saturated buckets
  - running out of samples in a bucket
  - mis-attributing phase residual to gain because lateralDelay is wrong
  - getting filtered out by gates that don't fire offline

The offline batch fit is just `BucketAccumulator.sum_residual / count` at the
end of the route, i.e., what the learner *would* converge to if its only
constraint was sample-mean.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUNNYPILOT_ROOT = Path.home() / "openpilot10"


def _ensure_path():
  if not (SUNNYPILOT_ROOT / "selfdrive" / "locationd" / "curvatured.py").exists():
    raise RuntimeError(
      f"sunnypilot learner source not found at {SUNNYPILOT_ROOT}. "
      f"Set SUNNYPILOT_ROOT or clone the dynamic_steering branch there."
    )
  p = str(SUNNYPILOT_ROOT)
  if p not in sys.path:
    sys.path.insert(0, p)


@dataclass
class LearnerReplayResult:
  converged_bias: np.ndarray              # shape (7, 12), learner's converged per-bucket bias
  bucket_counts: np.ndarray               # shape (7, 12), learner's per-bucket sample counts
  calibration_percent: int
  n_measurements_fed: int
  saturated_buckets: int                  # buckets where |bias| reached the +/-50% relative cap


def replay(dongle_id: str, route_id: str) -> LearnerReplayResult:
  """Replay the route through CurvatureEstimator.add_measurement on the same gated samples
  the extractor would use. Useful for cross-checking against the offline batch fit."""
  _ensure_path()
  from selfdrive.locationd.curvatured import CurvatureEstimator, CurvatureDLookup
  from cereal import car
  from opendbc.car.volkswagen.values import CAR

  from .extract import EXPECTED_FINGERPRINT
  from .gates import GateState
  from .signals import iter_samples

  CP = car.CarParams.new_message()
  CP.carFingerprint = CAR.VOLKSWAGEN_ID4_MK1
  CP.brand = "volkswagen"
  CP.steerControlType = car.CarParams.SteerControlType.curvatureDEPRECATED
  estimator = CurvatureEstimator(CP)
  estimator.use_params = True             # bypass the params toggle so we can fold measurements directly

  gate = GateState()
  n_fed = 0
  for sample in iter_samples(f"{dongle_id}/{route_id}/a"):
    if not sample.lat_active:
      gate.note_lat_inactive(sample.t)
    if sample.steering_pressed:
      gate.note_override(sample.t)
    if gate.check(sample.t, sample.lat_active, sample.v_ego, sample.desired_curvature,
                  sample.roll, sample.yaw_rate_std, sample.pose_valid) is not None:
      continue
    actual = sample.yaw_rate / max(sample.v_ego, 0.1) - sample.roll_compensation
    estimator.add_measurement(sample.desired_curvature, actual, sample.v_ego, schedule_only=False)
    n_fed += 1

  bias = np.asarray(estimator.bias, dtype=np.float64)
  counts = np.asarray(estimator.counts, dtype=np.float64)

  # detect saturated buckets: |bias| >= 0.5*|center| (i.e., bias hit the relative learning cap)
  centers = np.asarray(CurvatureDLookup.CURVATURE_BUCKET_CENTERS, dtype=np.float64)[None, :]
  saturated = int(np.sum(np.abs(bias) >= 0.5 * centers * 0.999))

  cal_pct = int(CurvatureDLookup.calibration_percent(counts))

  return LearnerReplayResult(
    converged_bias=bias,
    bucket_counts=counts,
    calibration_percent=cal_pct,
    n_measurements_fed=n_fed,
    saturated_buckets=saturated,
  )
