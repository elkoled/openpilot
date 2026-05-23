#!/usr/bin/env python3
"""
Stage 2 — Per-segment feature extraction.

Walks one segment's rlog, time-aligns the relevant cereal messages and CAN
signals at the segment's native 100 Hz cadence (one row per carControl frame),
and emits a single SegmentRecord with:

  - per-bucket aggregates (learner-supported 7x12 grid + extended 7x16 grid),
    computed twice: once gated by the openpilot10 learner's gates, once
    ungated. Ungated lets us see where residual lives outside the learner's
    support region — one of the explicit open questions.

  - per-segment scalars: median/p95 highway residual, per-speed-bin steady-
    state gain in (0.3, 1.7) (out-of-bounds excluded, not silently reported),
    cross-correlation lag, engaged seconds, etc.

  - learner replay result (final bias + counts) using the same in-memory
    samples we already have.

  - batch-fit per-bucket result using the same samples.

By design we hold one segment's per-frame arrays only inside this function;
on return we hand back the compact record (a few KB) — never raw 100 Hz
timelines, never lists of segments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np

from openpilot.tools.vw_id4_lateral.grid import (
  BucketAccumulator, EXTENDED_SHAPE, LEARNER_SHAPE,
  EXTENDED_CURVATURE_EDGES, EXTENDED_SPEED_ANCHORS,
  LEARNER_CURVATURE_EDGES, LEARNER_SPEED_ANCHORS,
  curvature_bucket, speed_bucket,
)
from openpilot.tools.vw_id4_lateral.learner_replay import (
  CurvatureDLookup, OfflineCurvatureEstimator, batch_fit_per_bucket,
)

G = 9.80665
DT = 0.01  # carControl is 100 Hz on MEB
HIGHWAY_MIN_KPH = 80.0
HIGHWAY_MAX_KPH = 130.0
ENGAGE_BUFFER_S = 2.0
OVERRIDE_BUFFER_S = 2.0
MAX_YAW_RATE_STD = 1.0
MAX_LEARN_LAT_ACCEL = 1.0
MAX_LEARN_ROLL_LAT_ACCEL = 0.10
DBC_NAME = "vw_meb"


@dataclass
class SegmentRecord:
  route: str
  dongle: str
  fingerprint: str
  vin: str
  build_year: str
  steer_control_type: str
  ok: bool = False
  reason: str = ""

  # raw timing
  duration_s: float = 0.0
  engaged_s: float = 0.0
  gated_samples: int = 0
  ungated_samples: int = 0

  # per-segment scalars
  highway_abs_residual_p50: float = float("nan")
  highway_abs_residual_p95: float = float("nan")
  per_speed_bin_gain: dict = field(default_factory=dict)        # {"80kph": 0.97, ...}
  per_speed_bin_gain_n: dict = field(default_factory=dict)
  delay_xcorr_lag_s: float = float("nan")

  # live params summaries
  mean_steer_ratio: float = float("nan")
  mean_stiffness_factor: float = float("nan")
  mean_lateral_delay: float = float("nan")
  mean_eps_power: float = float("nan")

  # bucket aggregates (serialized via .to_dict)
  buckets_learner_gated: dict = field(default_factory=dict)
  buckets_learner_ungated: dict = field(default_factory=dict)
  buckets_extended_ungated: dict = field(default_factory=dict)

  # CAN-checked vs cereal-reported divergence
  curvature_meas_can_minus_cereal_p95: float = float("nan")

  # learner replay output (gated samples only)
  learner_replay: dict = field(default_factory=dict)
  # offline batch fit on same gated samples
  batch_fit: dict = field(default_factory=dict)

  # PID-on flag from upstream
  has_live_curvature_parameters: bool = False
  # sidecar timeline cache path if written (used by plant_fit)
  timeline_cache_path: str = ""


def _detect_qfk_bus(lr) -> int:
  """First-pass scan: which src bus carries QFK_01 (addr 317)? Common buses
  are 0 and 2 across the fleet."""
  counts: dict[int, int] = {}
  for i, msg in enumerate(lr):
    if msg.which() != "can":
      continue
    for f in msg.can:
      if f.address == 317:  # QFK_01
        counts[f.src] = counts.get(f.src, 0) + 1
    if i > 2000 and counts:
      break
  if not counts:
    return 0
  return max(counts, key=counts.get)


def _logreader_identifier(route: str, rlog_url: str) -> str:
  """If rlog_url is provided (CSV-driven path) use it verbatim. Otherwise
  treat route as a canonical dongle/route_id and append /r for RLOG."""
  if rlog_url:
    return rlog_url
  return f"{route}/r"


def extract(route: str, dongle: str = "", fingerprint: str = "",
            vin: str = "", build_year: str = "",
            steer_control_type: str = "",
            has_live_curvature_parameters: bool = False,
            rlog_url: str = "",
            timeline_cache_dir: str = "") -> SegmentRecord:
  """Process one route (or one segment). Returns SegmentRecord.

  Either pass a canonical route id (`dongle/route_id`) and the function will
  resolve it via openpilot's LogReader sources, OR pass `rlog_url` for direct
  URL access (e.g. comma's data-fallback endpoint). The two paths produce
  identical SegmentRecords."""
  from openpilot.tools.lib.logreader import LogReader, ReadMode
  from openpilot.selfdrive.pandad import can_capnp_to_list
  from opendbc.can import CANParser

  rec = SegmentRecord(route=route, dongle=dongle or (route.split("/")[0] if "/" in route else ""),
                      fingerprint=fingerprint, vin=vin, build_year=build_year,
                      steer_control_type=steer_control_type,
                      has_live_curvature_parameters=has_live_curvature_parameters)

  identifier = _logreader_identifier(route, rlog_url)
  try:
    bus_probe_lr = LogReader(identifier, default_mode=ReadMode.RLOG)
    qfk_bus = _detect_qfk_bus(bus_probe_lr)
  except Exception as e:
    rec.reason = f"open_failed:{type(e).__name__}:{str(e)[:120]}"
    return rec

  # Per-frame staging arrays. Reset on engagement edges to avoid mixing
  # different engagement spans. We size growable lists and convert to numpy
  # at the end.
  t = []
  v_ego = []
  yaw_rate = []
  yaw_std = []
  roll = []
  desired_curv_cc = []        # carControl.actuators.curvature
  current_curv_cc = []        # carControl.currentCurvature  (post-controller)
  curv_meas_cs = []           # carState.curvatureMeas  (rebuilt from cereal)
  curv_meas_can = []          # CAN-decoded QFK_01.Curvature
  hca_curv_can = []           # CAN-decoded HCA_03.Curvature
  hca_status_can = []         # HCA_03.RequestStatus
  eps_power_can = []          # HCA_03.Power
  lat_active = []
  steering_pressed = []
  steer_ratio = []
  stiffness = []
  lat_delay = []

  try:
    lr = LogReader(identifier, default_mode=ReadMode.RLOG)
    cp = CANParser(DBC_NAME, [("QFK_01", 0), ("HCA_03", 0), ("LH_EPS_03", 0)], qfk_bus)

    # Running latest values
    cur = {
      "vEgo": float("nan"), "yawRate": float("nan"), "yawStd": float("nan"),
      "roll": 0.0, "steeringPressed": False, "latActive": False,
      "actuatorsCurvature": float("nan"), "currentCurvature": float("nan"),
      "curvatureMeas": float("nan"),
      "steerRatio": float("nan"), "stiffness": float("nan"),
      "lateralDelay": float("nan"),
      "qfkCurv": float("nan"), "hcaCurv": float("nan"),
      "hcaStatus": float("nan"), "epsPower": float("nan"),
    }
    t0 = None
    has_curvature_meas_field = True

    for msg in lr:
      which = msg.which()
      mt = msg.logMonoTime * 1e-9
      if t0 is None:
        t0 = mt

      if which == "can":
        cp.update(can_capnp_to_list([msg.as_builder().to_bytes()]))
        # signed QFK curvature
        qfk_vz = int(cp.vl["QFK_01"]["Curvature_VZ"])
        qfk_c = float(cp.vl["QFK_01"]["Curvature"]) * (1 if qfk_vz == 0 else -1)
        # carstate.py negates the sign — see opendbc.car.volkswagen.carstate.py:257
        cur["qfkCurv"] = -qfk_c
        hca_vz = int(cp.vl["HCA_03"]["Curvature_VZ"])
        hca_c = float(cp.vl["HCA_03"]["Curvature"]) * (1 if hca_vz == 1 else -1)
        cur["hcaCurv"] = hca_c
        cur["hcaStatus"] = float(cp.vl["HCA_03"]["RequestStatus"])
        cur["epsPower"] = float(cp.vl["HCA_03"]["Power"])

      elif which == "carState":
        cs = msg.carState
        cur["vEgo"] = float(cs.vEgo)
        cur["steeringPressed"] = bool(cs.steeringPressed)
        if has_curvature_meas_field:
          try:
            cur["curvatureMeas"] = float(cs.curvatureMeas)
          except AttributeError:
            has_curvature_meas_field = False

      elif which == "carControl":
        cc = msg.carControl
        cur["latActive"] = bool(cc.latActive)
        cur["actuatorsCurvature"] = float(cc.actuators.curvature)
        try:
          cur["currentCurvature"] = float(cc.currentCurvature)
        except Exception:
          cur["currentCurvature"] = float("nan")
        # Sample at carControl cadence (100 Hz)
        t.append(mt - t0)
        v_ego.append(cur["vEgo"])
        yaw_rate.append(cur["yawRate"])
        yaw_std.append(cur["yawStd"])
        roll.append(cur["roll"])
        desired_curv_cc.append(cur["actuatorsCurvature"])
        current_curv_cc.append(cur["currentCurvature"])
        curv_meas_cs.append(cur["curvatureMeas"])
        curv_meas_can.append(cur["qfkCurv"])
        hca_curv_can.append(cur["hcaCurv"])
        hca_status_can.append(cur["hcaStatus"])
        eps_power_can.append(cur["epsPower"])
        lat_active.append(cur["latActive"])
        steering_pressed.append(cur["steeringPressed"])
        steer_ratio.append(cur["steerRatio"])
        stiffness.append(cur["stiffness"])
        lat_delay.append(cur["lateralDelay"])

      elif which == "livePose":
        lp = msg.livePose
        try:
          cur["yawRate"] = float(lp.angularVelocityDevice.z)
          cur["yawStd"] = float(lp.angularVelocityDevice.zStd)
        except Exception:
          pass
        try:
          cur["roll"] = float(lp.orientationNED.x)
        except Exception:
          pass

      elif which == "liveParameters":
        lpa = msg.liveParameters
        try:
          cur["steerRatio"] = float(lpa.steerRatio)
        except Exception:
          pass
        try:
          cur["stiffness"] = float(lpa.stiffnessFactor)
        except Exception:
          pass

      elif which == "liveDelay":
        try:
          cur["lateralDelay"] = float(msg.liveDelay.lateralDelay)
        except Exception:
          pass

  except Exception as e:
    rec.reason = f"iterate_failed:{type(e).__name__}:{str(e)[:120]}"
    return rec

  n = len(t)
  if n < 200:
    rec.reason = f"too_few_samples:{n}"
    return rec

  # Convert to arrays
  t_arr = np.asarray(t, dtype=np.float64)
  v = np.asarray(v_ego, dtype=np.float64)
  yr = np.asarray(yaw_rate, dtype=np.float64)
  ys = np.asarray(yaw_std, dtype=np.float64)
  rl = np.asarray(roll, dtype=np.float64)
  desired = np.asarray(desired_curv_cc, dtype=np.float64)
  current = np.asarray(current_curv_cc, dtype=np.float64)
  cmeas = np.asarray(curv_meas_cs, dtype=np.float64)
  cmeas_can = np.asarray(curv_meas_can, dtype=np.float64)
  hca_curv = np.asarray(hca_curv_can, dtype=np.float64)
  hca_status = np.asarray(hca_status_can, dtype=np.float64)
  power = np.asarray(eps_power_can, dtype=np.float64)
  active = np.asarray(lat_active, dtype=bool)
  pressed = np.asarray(steering_pressed, dtype=bool)
  sr = np.asarray(steer_ratio, dtype=np.float64)
  sf = np.asarray(stiffness, dtype=np.float64)
  ld = np.asarray(lat_delay, dtype=np.float64)

  rec.duration_s = float(t_arr[-1] - t_arr[0])
  rec.engaged_s = float(np.sum(active) * DT)

  # Actual curvature: kinematic yaw_rate / vEgo. We do NOT subtract a
  # roll-bank "compensation" here — on a banked road driving straight,
  # yaw_rate is already zero, so the kinematic curvature is zero. The
  # openpilot10 daemon subtracts a `rollCompensation` signal that's a
  # fork-specific carControl field representing the extra curvature
  # command needed to hold the vehicle on the bank; that's a controller
  # internal, not a kinematic correction, and it's not present in stock
  # openpilot logs. We rely on the |sin(roll)·g| ≤ 0.10 m/s² roll gate
  # below to exclude banked-road frames from the learner samples.
  with np.errstate(invalid="ignore", divide="ignore"):
    actual = yr / np.maximum(v, 0.1)

  # Apply lateral delay to desired→actual alignment. Prefer liveDelay if the
  # log carries it; otherwise estimate it per-segment from cross-correlation
  # of desired vs actual at highway speed. No guessed defaults — if neither
  # is available, no shift is applied and the residual carries lag content.
  median_delay = float("nan")
  if np.any(~np.isnan(ld)):
    median_delay = float(np.nanmedian(ld))
  else:
    # data-driven lag estimate: argmax xcorr at highway speed, engaged frames
    hwy = active & np.isfinite(desired) & np.isfinite(actual) & \
          (v >= HIGHWAY_MIN_KPH / 3.6) & (v <= HIGHWAY_MAX_KPH / 3.6)
    if int(np.sum(hwy)) > 1000:
      d_h = desired[hwy] - np.mean(desired[hwy])
      a_h = actual[hwy] - np.mean(actual[hwy])
      max_lag = 30  # 300 ms
      best_lag = 0
      best_r = -np.inf
      d_var = float(np.sum(d_h * d_h))
      a_var = float(np.sum(a_h * a_h))
      norm = math.sqrt(d_var * a_var) if d_var > 0 and a_var > 0 else 0.0
      if norm > 0:
        for L in range(0, max_lag + 1):
          r = float(np.sum(d_h[:len(d_h) - L] * a_h[L:]))
          if r > best_r:
            best_r = r
            best_lag = L
        median_delay = best_lag * DT
  if math.isnan(median_delay):
    median_delay = 0.0
  shift_samples = int(round(median_delay / DT))
  shift_samples = max(0, min(shift_samples, 50))  # clamp to [0, 0.5s]
  if shift_samples > 0:
    desired_shifted = np.concatenate([np.full(shift_samples, np.nan), desired[:-shift_samples]])
    current_shifted = np.concatenate([np.full(shift_samples, np.nan), current[:-shift_samples]])
  else:
    desired_shifted = desired.copy()
    current_shifted = current.copy()
  rec.delay_xcorr_lag_s = median_delay

  # Engagement edges → engagement buffer mask
  engage_buffer = np.zeros(n, dtype=bool)
  last_inactive = -1e9
  for i in range(n):
    if not active[i]:
      last_inactive = t_arr[i]
    engage_buffer[i] = (t_arr[i] - last_inactive) >= ENGAGE_BUFFER_S

  override_buffer = np.zeros(n, dtype=bool)
  last_press = -1e9
  for i in range(n):
    if pressed[i]:
      last_press = t_arr[i]
    override_buffer[i] = (t_arr[i] - last_press) >= OVERRIDE_BUFFER_S

  # Learner gates (matches OfflineCurvatureEstimator constraints; this file
  # owns the temporal gates, the learner owns the bucket-grid gates).
  lat_accel_ok = (np.abs(desired_shifted) * v * v) <= MAX_LEARN_LAT_ACCEL
  roll_ok = np.abs(np.sin(rl) * G) <= MAX_LEARN_ROLL_LAT_ACCEL
  yaw_std_ok = ys < MAX_YAW_RATE_STD
  speed_ok = v >= float(LEARNER_SPEED_ANCHORS[0]) * 0.5
  finite_ok = np.isfinite(desired_shifted) & np.isfinite(actual) & np.isfinite(v)

  gated_mask = active & engage_buffer & override_buffer & lat_accel_ok & \
               roll_ok & yaw_std_ok & speed_ok & finite_ok
  ungated_engaged = active & finite_ok  # baseline for "where residual lives"

  rec.gated_samples = int(np.sum(gated_mask))
  rec.ungated_samples = int(np.sum(ungated_engaged))

  # Lower per-segment threshold than the openpilot10 daemon's online MIN
  # because we pool across many segments per car. A 60-second highway
  # segment with truly straight driving may legitimately have only a few
  # gated samples (most |desired_curvature| < 1e-6 falls outside the
  # learner's smallest bucket).
  if rec.gated_samples < 30:
    rec.reason = f"insufficient_gated_samples:{rec.gated_samples}"
    rec.ok = False
  else:
    rec.ok = True

  # Per-speed-bin steady-state gain (in low-curvature, low-lat-accel windows)
  per_speed_bin_gain = {}
  per_speed_bin_n = {}
  for v_anchor_kph in (40, 60, 80, 100, 120):
    v_anchor = v_anchor_kph / 3.6
    spd_mask = (np.abs(v - v_anchor) < 5.0 / 3.6) & gated_mask
    # narrow band: small lateral accel
    spd_mask &= (np.abs(desired_shifted) > 5e-5) & (np.abs(desired_shifted) < 1e-3)
    n_in = int(np.sum(spd_mask))
    if n_in < 100:
      continue
    a = actual[spd_mask]
    d = desired_shifted[spd_mask]
    # Weighted least-squares slope through origin: sum(d*a) / sum(d*d)
    gain = float(np.sum(d * a) / max(float(np.sum(d * d)), 1e-9))
    if 0.3 < gain < 1.7:
      per_speed_bin_gain[f"{v_anchor_kph}kph"] = gain
      per_speed_bin_n[f"{v_anchor_kph}kph"] = n_in
  rec.per_speed_bin_gain = per_speed_bin_gain
  rec.per_speed_bin_gain_n = per_speed_bin_n

  # Highway residual percentiles (engaged + speed in 80..130 kph, ungated
  # for the other criteria so we see the natural distribution).
  hwy_mask = active & finite_ok & (v >= HIGHWAY_MIN_KPH / 3.6) & (v <= HIGHWAY_MAX_KPH / 3.6)
  if np.sum(hwy_mask) > 100:
    res = desired_shifted[hwy_mask] - actual[hwy_mask]
    rec.highway_abs_residual_p50 = float(np.nanpercentile(np.abs(res), 50))
    rec.highway_abs_residual_p95 = float(np.nanpercentile(np.abs(res), 95))

  # liveParameters / liveDelay summaries
  rec.mean_steer_ratio = float(np.nanmean(sr)) if np.any(np.isfinite(sr)) else float("nan")
  rec.mean_stiffness_factor = float(np.nanmean(sf)) if np.any(np.isfinite(sf)) else float("nan")
  rec.mean_lateral_delay = float(np.nanmean(ld)) if np.any(np.isfinite(ld)) else float("nan")
  rec.mean_eps_power = float(np.nanmean(power[active])) if np.any(active) else float("nan")

  # CAN-vs-cereal sanity: how different is QFK_01 (CAN) from carState.curvatureMeas (cereal)?
  diff = cmeas_can - cmeas
  diff = diff[np.isfinite(diff)]
  if len(diff) > 100:
    rec.curvature_meas_can_minus_cereal_p95 = float(np.nanpercentile(np.abs(diff), 95))

  # ----- bucket aggregation (gated + ungated, learner grid + extended) -----
  def aggregate(mask: np.ndarray, edges: np.ndarray, anchors: np.ndarray,
                shape: tuple) -> BucketAccumulator:
    bucket = BucketAccumulator(shape)
    if not np.any(mask):
      return bucket
    d_sel = desired_shifted[mask]
    a_sel = actual[mask]
    v_sel = v[mask]
    r_sel = rl[mask]
    # vectorized bucket index assignment is cleaner but the per-bucket sums
    # need conditional left/right; loop is fine at 100Hz × ~minutes.
    for i in range(len(d_sel)):
      di = float(d_sel[i])
      si_idx = speed_bucket(float(v_sel[i]), anchors)
      ci_idx = curvature_bucket(di, edges)
      if si_idx is None or ci_idx is None:
        continue
      # learner uses direction-projected error: direction*(desired - actual)
      direction = 1.0 if di >= 0.0 else -1.0
      signed_err = direction * (di - float(a_sel[i]))
      sgn = 1 if di > 0 else (-1 if di < 0 else 0)
      bucket.add(si_idx, ci_idx, signed_err, sgn, float(r_sel[i]))
    return bucket

  rec.buckets_learner_gated = aggregate(gated_mask, LEARNER_CURVATURE_EDGES,
                                        LEARNER_SPEED_ANCHORS, LEARNER_SHAPE).to_dict()
  rec.buckets_learner_ungated = aggregate(active & finite_ok, LEARNER_CURVATURE_EDGES,
                                          LEARNER_SPEED_ANCHORS, LEARNER_SHAPE).to_dict()
  rec.buckets_extended_ungated = aggregate(active & finite_ok, EXTENDED_CURVATURE_EDGES,
                                           EXTENDED_SPEED_ANCHORS, EXTENDED_SHAPE).to_dict()

  # ----- learner replay (gated samples only) -----
  if rec.gated_samples >= 30:
    d_g = desired_shifted[gated_mask]
    a_g = actual[gated_mask]
    v_g = v[gated_mask]
    ys_g = ys[gated_mask]
    replay = OfflineCurvatureEstimator().replay(d_g, a_g, v_g)
    rec.learner_replay = {
      "bias": replay.bias.tolist(),
      "counts": replay.counts.tolist(),
      "calibration_percent": replay.calibration_percent,
      "num_applied": replay.num_samples_applied,
      "num_rejected": replay.num_samples_rejected,
    }
    batch = batch_fit_per_bucket(d_g, a_g, v_g, ys_g)
    rec.batch_fit = {
      "bias": batch.bias.tolist(),
      "bias_var": batch.bias_var.tolist(),
      "counts": batch.counts.tolist(),
      "bias_clipped_fraction": batch.bias_clipped_fraction.tolist(),
    }

  # ----- timeline cache (for plant_fit) -----
  # Save the engaged 100 Hz timeline as a sidecar npz under <cache>/<dongle>/
  # <safe_route>.npz . plant_fit reads these to do ARX(1,1,Td) fits which
  # need time-correlated samples, not just bucket aggregates.
  if timeline_cache_dir and rec.engaged_s >= 5.0:
    import os
    safe = route.replace("/", "__")
    car_dir = os.path.join(timeline_cache_dir, rec.dongle or "unknown")
    os.makedirs(car_dir, exist_ok=True)
    out_path = os.path.join(car_dir, f"{safe}.npz")
    eng_mask = active & np.isfinite(desired) & np.isfinite(actual) & np.isfinite(v)
    if int(np.sum(eng_mask)) > 100:
      np.savez_compressed(
        out_path,
        v=v[eng_mask].astype(np.float32),
        desired=desired[eng_mask].astype(np.float32),
        actual=actual[eng_mask].astype(np.float32),
        desired_shifted=desired_shifted[eng_mask].astype(np.float32),
        roll=rl[eng_mask].astype(np.float32),
        yaw_std=ys[eng_mask].astype(np.float32),
        steering_pressed=pressed[eng_mask],
        t=t_arr[eng_mask].astype(np.float32),
        median_delay_s=np.float32(median_delay),
      )
      rec.timeline_cache_path = out_path

  return rec


def record_to_dict(rec: SegmentRecord) -> dict:
  return asdict(rec)
