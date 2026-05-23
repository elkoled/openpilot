"""Per-route worker: streams a route through signals + gates + features and
emits a single fixed-size feature dict (+ status).

Designed for use inside multiprocessing.Pool — keep it pure (no shared state),
catch every exception, return a structured failure code on error.
"""
from __future__ import annotations

import math
import pickle
import time
import traceback
from dataclasses import dataclass, field

from .features import BucketAccumulator, XcorrAccumulator, ConditioningHistograms
from .gates import GateState
from .signals import iter_samples

EXPECTED_FINGERPRINT = "VOLKSWAGEN_ID4_MK1"

# (label, desired_getter, actual_getter). 'desired' is the lag-adjusted target carControl
# hands the controller; 'apply' is what carcontroller actually sends after its additive correction.
PAIRS = (
  ("P1_desired_yaw", lambda s: s.desired_curvature, lambda s: s.yaw_rate / max(s.v_ego, 0.1) - s.roll_compensation),
  ("P2_desired_qfk", lambda s: s.desired_curvature, lambda s: s.qfk_curvature),
  ("P3_apply_yaw",   lambda s: s.apply_curvature,   lambda s: s.yaw_rate / max(s.v_ego, 0.1) - s.roll_compensation),
  ("P4_apply_qfk",   lambda s: s.apply_curvature,   lambda s: s.qfk_curvature),
)


@dataclass
class ExtractResult:
  route_key: str
  dongle_id: str
  route_id: str
  branch: str
  status: str                         # 'ok' | 'fingerprint_mismatch' | 'no_engagement' | 'error' | ...
  error: str = ""
  car_fingerprint: str = ""
  car_vin: str = ""
  engaged_seconds: float = 0.0
  total_seconds: float = 0.0
  n_samples_total: int = 0
  n_samples_gated: int = 0
  elapsed_s: float = 0.0
  features: dict = field(default_factory=dict)   # pickled blob


def run(dongle_id: str, route_id: str, branch: str = "", rlog_url: str = "") -> ExtractResult:
  route_key = route_id if rlog_url else f"{dongle_id}/{route_id}"
  start = time.monotonic()
  result = ExtractResult(route_key=route_key, dongle_id=dongle_id, route_id=route_id,
                         branch=branch, status="error")
  try:
    spec = rlog_url if rlog_url else f"{dongle_id}/{route_id}/a"
    _run_inner(spec, result)
  except Exception:
    result.status = "error"
    result.error = traceback.format_exc(limit=4)
  result.elapsed_s = time.monotonic() - start
  return result


def _run_inner(spec: str, result: ExtractResult) -> None:
  gate = GateState()
  pair_buckets = {label: BucketAccumulator(label) for (label, _, _) in PAIRS}
  pair_xcorr = {label: XcorrAccumulator(label) for (label, _, _) in PAIRS}
  cond = ConditioningHistograms()

  first_t: float | None = None
  last_t: float | None = None
  prev_t: float | None = None
  engaged_total = 0.0
  active_total = 0.0
  MAX_TICK_DELTA = 1.0      # ignore gaps larger than this when summing route duration
  fingerprint_locked = False

  for sample in iter_samples(spec):
    if first_t is None:
      first_t = sample.t
    last_t = sample.t
    result.n_samples_total += 1

    if not fingerprint_locked and sample.car_fingerprint:
      fingerprint_locked = True
      result.car_fingerprint = sample.car_fingerprint
      result.car_vin = sample.car_vin
      if sample.car_fingerprint != EXPECTED_FINGERPRINT:
        result.status = "fingerprint_mismatch"
        result.error = f"got {sample.car_fingerprint!r}, expected {EXPECTED_FINGERPRINT!r}"
        return

    if prev_t is not None:
      dt = sample.t - prev_t
      if 0.0 < dt < MAX_TICK_DELTA:
        active_total += dt
        if sample.lat_active:
          engaged_total += dt
    prev_t = sample.t

    if not sample.lat_active:
      gate.note_lat_inactive(sample.t)
    if sample.steering_pressed:
      gate.note_override(sample.t)

    desired = sample.desired_curvature
    rejection = gate.check(
      t=sample.t,
      lat_active=sample.lat_active,
      v_ego=sample.v_ego,
      desired_curvature=desired,
      roll=sample.roll,
      yaw_rate_std=sample.yaw_rate_std,
      pose_valid=sample.pose_valid,
    )
    if rejection is not None:
      continue

    result.n_samples_gated += 1
    lat_accel = desired * sample.v_ego * sample.v_ego
    cond.add_sample(sample.v_ego, lat_accel)
    if not math.isnan(sample.steer_ratio):
      cond.add_steer_ratio(sample.steer_ratio)
    if not math.isnan(sample.stiffness_factor):
      cond.add_stiffness(sample.stiffness_factor)
    cond.add_lateral_delay(sample.lateral_delay)
    cond.add_eps_power(sample.hca_power)
    cond.add_driver_torque(sample.driver_torque_nm)

    for (label, dget, aget) in PAIRS:
      d = dget(sample)
      a = aget(sample)
      pair_buckets[label].add(d, a, sample.v_ego)
      pair_xcorr[label].add(sample.t, d, a)

  result.total_seconds = active_total
  result.engaged_seconds = engaged_total

  if result.n_samples_gated == 0:
    if result.status == "error":
      result.status = "no_engagement"
      result.error = "no gated samples"
    return

  features: dict = {}
  for label in pair_buckets:
    features.update(pair_buckets[label].to_dict())
    features.update(pair_xcorr[label].to_dict())
  features.update(cond.to_dict())
  features.update(gate.summary())
  result.features = features
  result.status = "ok"


def pack_features(features: dict) -> bytes:
  return pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_features(blob: bytes) -> dict:
  return pickle.loads(blob)
