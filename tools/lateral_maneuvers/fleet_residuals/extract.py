"""Per-route lateral residual extractor for VW MEB (ID4_MK1).

Builds a compact summary of residuals at each hop of the lateral control chain:

    modelV2.action.desiredCurvature  (raw model)
        ->  controlsState.desiredCurvature  (controller, lag adjusted)
        ->  carControl.actuators.curvature  (input to carcontroller)
        ->  HCA_03.Curvature  (on-wire EPS command)
        ->  QFK_01.Curvature  (EPS-measured rack curvature)
        ->  yaw_rate / vEgo  - roll_compensation  (plant response, vehicle-frame)

Four residual surfaces are bucketed on the same (7 speed x 12 |curvature|)
grid the sunnypilot dynamic_steering learner uses, with the sign of the
desired curvature carried as a third dimension so left/right asymmetry is
visible:

    R_model_yaw  = model_raw_desired - yaw_actual_curvature           (learner view)
    R_model_qfk  = model_raw_desired - qfk_meas                        (model vs rack)
    R_hca_qfk    = hca_cmd          - qfk_meas                         (EPS execution)
    R_hca_yaw    = hca_cmd          - yaw_actual_curvature             (EPS->plant)

A fifth diagnostic, R_smooth_loss = model_raw - actuators_curvature,
captures the controller-internal smoothing/lag-adjustment effect.

Memory footprint per route is bounded by the bucket grid, not the route
duration: time series are summarised then discarded.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

# Lazy: heavy openpilot imports happen inside extract_route so that --help
# and unit-style use don't pay for them.

VERSION = 1

# Bucket grid - matches dynamic_steering learner
SPEED_ANCHORS_KMH = np.array([20., 40., 60., 80., 100., 120., 140.], dtype=np.float32)
SPEED_ANCHORS = (SPEED_ANCHORS_KMH / 3.6).astype(np.float32)
CURV_EDGES = np.array([
    1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5,
    1.28e-4, 2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3,
], dtype=np.float64)
CURV_CENTERS = np.sqrt(CURV_EDGES[:-1] * CURV_EDGES[1:])
N_SPEED = len(SPEED_ANCHORS)
N_CURV = len(CURV_CENTERS)
N_SIGN = 2  # 0 = negative curvature, 1 = positive

MIN_SPEED_MS = float(SPEED_ANCHORS[0] * 0.5)
RESAMPLE_HZ = 20
DT_GRID = 1.0 / RESAMPLE_HZ
ENGAGE_BUFFER_S = 2.0
MAX_YAW_RATE_STD = 1.0
MAX_LEARN_ROLL_LAT_ACCEL = 0.10  # m/s^2
ACC_GRAVITY = 9.81
MAX_LAT_ACCEL_APPLY = 1.0  # m/s^2, learner's apply gate (we report buckets above this separately)

# CAN
DBC_NAME = "vw_meb"
HCA_03_ADDR = 771
QFK_01_ADDR = 317
LH_EPS_03_ADDR = 159

# Lag estimation
LAG_SEARCH_S = 0.4
SPEED_BANDS_KMH = ((30, 60), (60, 100), (100, 140))

RESIDUAL_NAMES = ("R_model_yaw", "R_model_qfk", "R_hca_qfk", "R_hca_yaw", "R_smooth_loss")


# ---------- bucket math ----------

def speed_index(v_ms: float) -> int | None:
  if v_ms < MIN_SPEED_MS:
    return None
  return int(np.argmin(np.abs(SPEED_ANCHORS - v_ms)))


def curvature_index(curv: float) -> int | None:
  a = abs(float(curv))
  if a < CURV_EDGES[0] or a > CURV_EDGES[-1]:
    return None
  i = int(np.searchsorted(CURV_EDGES, a, side='right') - 1)
  return min(max(i, 0), N_CURV - 1)


def sign_index(curv: float) -> int:
  return 1 if curv >= 0.0 else 0


# ---------- accumulator ----------

@dataclass
class ResidualAccum:
  """Sufficient stats per (speed, |curv|, sign) cell for each residual surface.

  Stored as float64; size is N_SPEED * N_CURV * N_SIGN * 4 floats per residual
  surface = 7*12*2*4 = 672 cells, trivial.
  """
  shape: tuple = field(default_factory=lambda: (N_SPEED, N_CURV, N_SIGN))
  count: dict[str, np.ndarray] = field(default_factory=dict)
  sum: dict[str, np.ndarray] = field(default_factory=dict)
  sum_sq: dict[str, np.ndarray] = field(default_factory=dict)
  sum_abs: dict[str, np.ndarray] = field(default_factory=dict)

  def __post_init__(self):
    for name in RESIDUAL_NAMES:
      self.count[name] = np.zeros(self.shape, dtype=np.int64)
      self.sum[name] = np.zeros(self.shape, dtype=np.float64)
      self.sum_sq[name] = np.zeros(self.shape, dtype=np.float64)
      self.sum_abs[name] = np.zeros(self.shape, dtype=np.float64)

  def add(self, name: str, s_idx: int, c_idx: int, sgn_idx: int, err: float) -> None:
    self.count[name][s_idx, c_idx, sgn_idx] += 1
    self.sum[name][s_idx, c_idx, sgn_idx] += err
    self.sum_sq[name][s_idx, c_idx, sgn_idx] += err * err
    self.sum_abs[name][s_idx, c_idx, sgn_idx] += abs(err)


# ---------- LogReader ingest ----------

@dataclass
class SignalSeries:
  """Time-indexed value series. Times are in seconds (mono)."""
  t: list = field(default_factory=list)
  v: list = field(default_factory=list)

  def append(self, t_s: float, v: Any) -> None:
    self.t.append(float(t_s))
    self.v.append(v)

  def arrays(self) -> tuple[np.ndarray, np.ndarray]:
    if not self.t:
      return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    t = np.asarray(self.t, dtype=np.float64)
    v = np.asarray(self.v, dtype=np.float64)
    # LogReader returns messages per-segment, which may not be globally
    # time-ordered. Sort here so downstream resampling is correct.
    order = np.argsort(t, kind='stable')
    return t[order], v[order]


@dataclass
class RouteRaw:
  """Container for everything we pulled out of LogReader in one pass."""
  car_params: Any = None
  car_params_bytes: bytes | None = None

  v_ego: SignalSeries = field(default_factory=SignalSeries)
  steering_angle: SignalSeries = field(default_factory=SignalSeries)
  steering_torque: SignalSeries = field(default_factory=SignalSeries)
  steering_pressed: SignalSeries = field(default_factory=SignalSeries)
  curvature_meas_carstate: SignalSeries = field(default_factory=SignalSeries)  # CS.steeringCurvature on MEB

  lat_active: SignalSeries = field(default_factory=SignalSeries)
  cc_actuator_curv: SignalSeries = field(default_factory=SignalSeries)  # carControl.actuators.curvature
  cc_current_curv: SignalSeries = field(default_factory=SignalSeries)   # carControl.currentCurvature
  cc_roll_comp: SignalSeries = field(default_factory=SignalSeries)      # carControl.rollCompensation
  override_t: SignalSeries = field(default_factory=SignalSeries)

  co_actuator_curv: SignalSeries = field(default_factory=SignalSeries)  # carOutput.actuatorsOutput.curvature

  cs_desired_curv: SignalSeries = field(default_factory=SignalSeries)   # controlsState.desiredCurvature (lag-adj)
  cs_curvature: SignalSeries = field(default_factory=SignalSeries)       # controlsState.curvature (vehicle model)

  model_raw_desired: SignalSeries = field(default_factory=SignalSeries)  # modelV2.action.desiredCurvature

  yaw_rate_dev: SignalSeries = field(default_factory=SignalSeries)
  yaw_rate_std: SignalSeries = field(default_factory=SignalSeries)
  roll_dev: SignalSeries = field(default_factory=SignalSeries)
  pose_inputs_ok: SignalSeries = field(default_factory=SignalSeries)
  pose_net_ok: SignalSeries = field(default_factory=SignalSeries)

  lp_steer_ratio: SignalSeries = field(default_factory=SignalSeries)
  lp_stiffness: SignalSeries = field(default_factory=SignalSeries)
  lp_angle_offset: SignalSeries = field(default_factory=SignalSeries)
  lp_roll: SignalSeries = field(default_factory=SignalSeries)

  ld_lat_delay: SignalSeries = field(default_factory=SignalSeries)
  ld_lat_delay_est: SignalSeries = field(default_factory=SignalSeries)

  live_calib_rpy: SignalSeries = field(default_factory=SignalSeries)  # latest rpyCalib snapshot

  hca_curv_signed: SignalSeries = field(default_factory=SignalSeries)
  hca_request_status: SignalSeries = field(default_factory=SignalSeries)
  hca_power: SignalSeries = field(default_factory=SignalSeries)
  qfk_curv_signed: SignalSeries = field(default_factory=SignalSeries)
  qfk_hca_status: SignalSeries = field(default_factory=SignalSeries)
  eps_torque: SignalSeries = field(default_factory=SignalSeries)
  eps_hca_status: SignalSeries = field(default_factory=SignalSeries)

  # Inferred:
  has_modelDesiredCurvature: bool = False        # sunnypilot extension presence
  has_liveCurvatureParameters: bool = False      # sunnypilot learner presence
  has_can: bool = False
  can_bus_used: int | None = None
  log_count_by_type: dict = field(default_factory=lambda: defaultdict(int))


def _safe_get(msg: Any, *path: str, default: Any = None) -> Any:
  cur = msg
  for p in path:
    if cur is None:
      return default
    try:
      cur = getattr(cur, p)
    except Exception:
      return default
  return cur


def collect_route(route_id: str, max_seconds: float | None = None) -> RouteRaw:
  """Single pass through LogReader. Builds RouteRaw with timestamped series."""
  from openpilot.tools.lib.logreader import LogReader

  raw = RouteRaw()

  lr = LogReader(route_id)

  # Track candidate bus packet counts for HCA_03 / QFK_01 separately so we
  # can pick a single bus that has both.
  bus_hca_count: dict[int, int] = defaultdict(int)
  bus_qfk_count: dict[int, int] = defaultdict(int)

  # We need a two-pass strategy for CAN: first pass collects per-bus counts
  # for the two addresses we care about, then we re-iterate and parse. But
  # LogReader is iterable not seekable in general, so cache (t, packets) on
  # the fly to avoid a re-iteration.
  can_buf: list[tuple[int, list[tuple[int, bytes, int]]]] = []
  send_buf: list[tuple[int, list[tuple[int, bytes, int]]]] = []

  t0 = None
  for msg in lr:
    which = msg.which()
    raw.log_count_by_type[which] += 1
    t_ns = msg.logMonoTime
    t_s = t_ns * 1e-9
    if t0 is None:
      t0 = t_s
    if max_seconds is not None and (t_s - t0) > max_seconds:
      break

    if which == "carParams":
      cp = msg.carParams
      # Eagerly extract scalar fields we need; raw capnp readers don't
      # survive past LogReader iteration.
      raw.car_params = type('CP', (), {
        'carFingerprint': str(getattr(cp, 'carFingerprint', '')),
        'brand': str(getattr(cp, 'brand', '')),
        'carVin': str(getattr(cp, 'carVin', '')),
        'flags': int(getattr(cp, 'flags', 0)),
        'steerControlType': str(getattr(cp, 'steerControlType', '')),
        'carFw': list(getattr(cp, 'carFw', []) or []),
      })()

    elif which == "carState":
      cs = msg.carState
      raw.v_ego.append(t_s, float(cs.vEgo))
      raw.steering_angle.append(t_s, float(cs.steeringAngleDeg))
      raw.steering_torque.append(t_s, float(cs.steeringTorque))
      raw.steering_pressed.append(t_s, 1.0 if bool(cs.steeringPressed) else 0.0)
      sc = float(getattr(cs, 'steeringCurvature', 0.0))
      raw.curvature_meas_carstate.append(t_s, sc)

    elif which == "carControl":
      cc = msg.carControl
      raw.lat_active.append(t_s, 1.0 if bool(cc.latActive) else 0.0)
      act = getattr(cc, 'actuators', None)
      ac = float(getattr(act, 'curvature', 0.0)) if act is not None else 0.0
      raw.cc_actuator_curv.append(t_s, ac)
      raw.cc_current_curv.append(t_s, float(getattr(cc, 'currentCurvature', 0.0)))
      raw.cc_roll_comp.append(t_s, float(getattr(cc, 'rollCompensation', 0.0)))

    elif which == "carOutput":
      co = msg.carOutput
      ao = getattr(co, 'actuatorsOutput', None)
      ov = float(getattr(ao, 'curvature', 0.0)) if ao is not None else 0.0
      raw.co_actuator_curv.append(t_s, ov)

    elif which == "controlsState":
      cs = msg.controlsState
      dc = float(getattr(cs, 'desiredCurvature', 0.0))
      raw.cs_desired_curv.append(t_s, dc)
      raw.cs_curvature.append(t_s, float(getattr(cs, 'curvature', 0.0)))
      # sunnypilot extension
      mdc = getattr(cs, 'modelDesiredCurvature', None)
      if mdc is not None and mdc != 0.0:
        raw.has_modelDesiredCurvature = True

    elif which == "modelV2":
      mv = msg.modelV2
      act = getattr(mv, 'action', None)
      if act is not None:
        raw.model_raw_desired.append(t_s, float(getattr(act, 'desiredCurvature', 0.0)))

    elif which == "livePose":
      lp = msg.livePose
      av = getattr(lp, 'angularVelocityDevice', None)
      if av is not None:
        # device-frame yaw = z; will be calibrated downstream if liveCalibration is present
        raw.yaw_rate_dev.append(t_s, float(av.z))
        std = float(getattr(av, 'zStd', 0.0)) if hasattr(av, 'zStd') else 0.0
        raw.yaw_rate_std.append(t_s, std)
      orient = getattr(lp, 'orientationNED', None)
      if orient is not None:
        raw.roll_dev.append(t_s, float(orient.x))  # roll (NED phi)
      raw.pose_inputs_ok.append(t_s, 1.0 if bool(getattr(lp, 'inputsOK', False)) else 0.0)
      raw.pose_net_ok.append(t_s, 1.0 if bool(getattr(lp, 'posenetOK', False)) else 0.0)

    elif which == "liveParameters":
      lp = msg.liveParameters
      raw.lp_steer_ratio.append(t_s, float(getattr(lp, 'steerRatio', 0.0)))
      raw.lp_stiffness.append(t_s, float(getattr(lp, 'stiffnessFactor', 0.0)))
      raw.lp_angle_offset.append(t_s, float(getattr(lp, 'angleOffsetDeg', 0.0)))
      raw.lp_roll.append(t_s, float(getattr(lp, 'roll', 0.0)))

    elif which == "liveDelay":
      ld = msg.liveDelay
      raw.ld_lat_delay.append(t_s, float(getattr(ld, 'lateralDelay', 0.0)))
      raw.ld_lat_delay_est.append(t_s, float(getattr(ld, 'lateralDelayEstimate', 0.0)))

    elif which == "liveCalibration":
      lc = msg.liveCalibration
      rpy = getattr(lc, 'rpyCalib', None)
      if rpy is not None and len(rpy) == 3:
        raw.live_calib_rpy.append(t_s, tuple(float(x) for x in rpy))

    elif which == "liveCurvatureParameters":
      raw.has_liveCurvatureParameters = True
      # Don't try to parse the surface here; flag presence only.

    elif which == "can":
      raw.has_can = True
      pkts = []
      for p in msg.can:
        try:
          addr = int(p.address)
          src = int(p.src)
          dat = bytes(p.dat)
        except Exception:
          continue
        if addr == QFK_01_ADDR:
          bus_qfk_count[src] += 1
        if addr in (HCA_03_ADDR, QFK_01_ADDR, LH_EPS_03_ADDR):
          pkts.append((addr, dat, src))
      if pkts:
        can_buf.append((t_ns, pkts))

    elif which == "sendcan":
      # HCA_03 commanded by openpilot. sendcan packets sometimes carry a
      # different src than the bus they'll be transmitted on; we'll count
      # per-src and pick the dominant one.
      pkts = []
      for p in msg.sendcan:
        try:
          addr = int(p.address)
          src = int(p.src)
          dat = bytes(p.dat)
        except Exception:
          continue
        if addr == HCA_03_ADDR:
          bus_hca_count[src] += 1
          pkts.append((addr, dat, src))
      if pkts:
        send_buf.append((t_ns, pkts))

  # Decide best buses (HCA from sendcan, QFK from can; they may live on
  # different src values).
  hca_bus = max(bus_hca_count, key=bus_hca_count.get, default=None) if bus_hca_count else None
  qfk_bus = max(bus_qfk_count, key=bus_qfk_count.get, default=None) if bus_qfk_count else None
  raw.can_bus_used = (hca_bus, qfk_bus)
  if qfk_bus is not None:
    _parse_can_bus(can_buf, qfk_bus, raw, parse_hca=False, parse_qfk=True)
  if hca_bus is not None:
    _parse_can_bus(send_buf, hca_bus, raw, parse_hca=True, parse_qfk=False)

  return raw


def _parse_can_bus(buf: list, bus: int, raw: RouteRaw,
                   parse_hca: bool, parse_qfk: bool) -> None:
  """Run a CANParser on a single bus to extract HCA_03 and/or QFK_01/LH_EPS_03.

  Sign convention: carstate uses -Curvature * (1,-1)[VZ] for QFK_01, so VZ=1
  flips the sign. We apply the same to HCA_03 so the two are comparable.
  """
  from opendbc.can.parser import CANParser

  cp = CANParser(DBC_NAME, [('HCA_03', 50), ('QFK_01', 50), ('LH_EPS_03', 100)], bus)

  # buf may not be globally time-sorted across segments
  buf_sorted = sorted(buf, key=lambda x: x[0])

  for t_ns, pkts in buf_sorted:
    updated = cp.update([(t_ns, pkts)])
    if parse_hca and HCA_03_ADDR in updated:
      d = cp.vl['HCA_03']
      vz = int(d['Curvature_VZ'])
      curv = float(d['Curvature']) * (1.0 if vz == 0 else -1.0)
      raw.hca_curv_signed.append(t_ns * 1e-9, -curv)
      raw.hca_request_status.append(t_ns * 1e-9, float(d.get('RequestStatus', 0)))
      raw.hca_power.append(t_ns * 1e-9, float(d.get('Power', 0)))
    if parse_qfk and QFK_01_ADDR in updated:
      d = cp.vl['QFK_01']
      vz = int(d['Curvature_VZ'])
      curv = float(d['Curvature']) * (1.0 if vz == 0 else -1.0)
      raw.qfk_curv_signed.append(t_ns * 1e-9, -curv)
      raw.qfk_hca_status.append(t_ns * 1e-9, float(d.get('LatCon_HCA_Status', 0)))
    if parse_qfk and LH_EPS_03_ADDR in updated:
      d = cp.vl['LH_EPS_03']
      raw.eps_torque.append(t_ns * 1e-9, float(d.get('EPS_Lenkmoment', 0)) * 0.01)
      raw.eps_hca_status.append(t_ns * 1e-9, float(d.get('EPS_HCA_Status', 0)))


# ---------- resampling ----------

def resample_last_value(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
  """Zero-order-hold resampling. Out-of-range left fills with the first value
  if available, else 0."""
  out = np.zeros_like(t_grid, dtype=np.float64)
  if t_src.size == 0:
    return out
  idx = np.searchsorted(t_src, t_grid, side='right') - 1
  idx = np.clip(idx, 0, len(t_src) - 1)
  out[:] = v_src[idx]
  out[t_grid < t_src[0]] = v_src[0]
  return out


def build_grid(raw: RouteRaw) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Build a 20Hz grid spanning the engaged-spans envelope and resample.

  Returns (t_grid, signals dict).
  """
  t_lat, v_lat = raw.lat_active.arrays()
  t_ve, v_ve = raw.v_ego.arrays()
  if t_lat.size == 0 or t_ve.size == 0:
    return np.zeros(0), {}

  t_start = max(float(t_lat[0]), float(t_ve[0]))
  t_end = min(float(t_lat[-1]), float(t_ve[-1]))
  if t_end <= t_start:
    return np.zeros(0), {}

  t_grid = np.arange(t_start, t_end, DT_GRID, dtype=np.float64)

  signals: dict[str, np.ndarray] = {}
  series_map = {
    'v_ego': raw.v_ego,
    'steering_angle': raw.steering_angle,
    'steering_torque': raw.steering_torque,
    'steering_pressed': raw.steering_pressed,
    'curvature_meas_cs': raw.curvature_meas_carstate,
    'lat_active': raw.lat_active,
    'cc_actuator_curv': raw.cc_actuator_curv,
    'cc_current_curv': raw.cc_current_curv,
    'cc_roll_comp': raw.cc_roll_comp,
    'co_actuator_curv': raw.co_actuator_curv,
    'cs_desired_curv': raw.cs_desired_curv,
    'cs_curvature': raw.cs_curvature,
    'model_raw_desired': raw.model_raw_desired,
    'yaw_rate_dev': raw.yaw_rate_dev,
    'yaw_rate_std': raw.yaw_rate_std,
    'roll_dev': raw.roll_dev,
    'pose_inputs_ok': raw.pose_inputs_ok,
    'pose_net_ok': raw.pose_net_ok,
    'lp_steer_ratio': raw.lp_steer_ratio,
    'lp_stiffness': raw.lp_stiffness,
    'lp_angle_offset': raw.lp_angle_offset,
    'lp_roll': raw.lp_roll,
    'ld_lat_delay': raw.ld_lat_delay,
    'ld_lat_delay_est': raw.ld_lat_delay_est,
    'hca_curv_signed': raw.hca_curv_signed,
    'hca_request_status': raw.hca_request_status,
    'hca_power': raw.hca_power,
    'qfk_curv_signed': raw.qfk_curv_signed,
    'qfk_hca_status': raw.qfk_hca_status,
    'eps_torque': raw.eps_torque,
    'eps_hca_status': raw.eps_hca_status,
  }
  for name, series in series_map.items():
    t_s, v_s = series.arrays()
    signals[name] = resample_last_value(t_s, v_s, t_grid)

  return t_grid, signals


# ---------- engagement gates ----------

def build_engagement_mask(t_grid: np.ndarray, signals: dict) -> tuple[np.ndarray, np.ndarray]:
  """Return (apply_mask, learn_mask).

  apply_mask: latActive, not pressed, v >= MIN_SPEED, pose inputs/net OK
  learn_mask: apply_mask AND yaw_std < 1, |roll*g| < 0.10, with the
              2-second buffer after override and after re-engagement.
  """
  if t_grid.size == 0:
    return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

  lat_active = signals['lat_active'] > 0.5
  pressed = signals['steering_pressed'] > 0.5
  v_ok = signals['v_ego'] >= MIN_SPEED_MS
  pose_ok = (signals['pose_inputs_ok'] > 0.5) & (signals['pose_net_ok'] > 0.5)

  apply_mask = lat_active & (~pressed) & v_ok & pose_ok

  # learn gates
  yaw_std_ok = signals['yaw_rate_std'] < MAX_YAW_RATE_STD
  roll_ok = np.abs(np.sin(signals['roll_dev']) * ACC_GRAVITY) <= MAX_LEARN_ROLL_LAT_ACCEL

  # 2s buffer after lat inactive rising edge and after override
  buf_steps = int(round(ENGAGE_BUFFER_S / DT_GRID))
  active_buf = np.zeros_like(lat_active, dtype=bool)
  press_buf = np.zeros_like(pressed, dtype=bool)
  # samples within buf_steps after a !active or pressed sample are NOT yet learnable
  inactive = (~lat_active) | pressed
  forbid = np.zeros_like(lat_active, dtype=bool)
  # forward-pass: mark forbid[i+1..i+buf_steps] when inactive[i]
  # Approximation: dilate inactive forward by buf_steps via cumulative trick
  if buf_steps > 0:
    cum = np.zeros(len(inactive) + buf_steps + 1, dtype=np.int32)
    idx = np.where(inactive)[0]
    if idx.size:
      cum[idx + 1] += 1
      cum[np.minimum(idx + buf_steps + 1, len(cum) - 1)] -= 1
      runs = np.cumsum(cum)[: len(inactive)]
      forbid = runs > 0
  learn_mask = apply_mask & yaw_std_ok & roll_ok & (~forbid)
  return apply_mask, learn_mask


# ---------- residual computation ----------

def compute_yaw_actual_curvature(signals: dict) -> np.ndarray:
  """Yaw-rate-derived plant curvature minus an approximate road-roll term.

  carControl.rollCompensation is deprecated; we synthesise the equivalent
  from liveParameters.roll using the dominant-term VM formula
  (slip-factor correction is <5%% at MEB highway speeds).
  """
  v = np.maximum(signals['v_ego'], 0.1)
  roll_comp = signals['lp_roll'] * ACC_GRAVITY / (v * v)
  return signals['yaw_rate_dev'] / v - roll_comp


def shift_forward(arr: np.ndarray, shift_steps: int) -> np.ndarray:
  """Return arr shifted forward in time by shift_steps. New leading samples
  copy arr[0]; loses shift_steps from the end (replaced with arr[-1])."""
  if shift_steps == 0 or arr.size == 0:
    return arr
  out = np.empty_like(arr)
  if shift_steps > 0:
    out[:shift_steps] = arr[0]
    out[shift_steps:] = arr[: len(arr) - shift_steps]
  else:
    s = -shift_steps
    out[: len(arr) - s] = arr[s:]
    out[len(arr) - s :] = arr[-1]
  return out


def median_lat_delay(signals: dict, mask: np.ndarray) -> float:
  d = signals['ld_lat_delay'][mask]
  d = d[d > 0]
  if d.size == 0:
    return 0.15  # plausible MEB default if no liveDelay was published
  return float(np.median(d))


def accumulate_residuals(signals: dict, learn_mask: np.ndarray, lat_delay_s: float,
                         require_hca: bool, require_qfk: bool) -> ResidualAccum:
  acc = ResidualAccum()
  if learn_mask.sum() == 0:
    return acc

  shift_steps = int(round(lat_delay_s / DT_GRID))
  yaw_curv = compute_yaw_actual_curvature(signals)

  # Past targets that should be reflected in current actual.
  model_raw_past = shift_forward(signals['model_raw_desired'], shift_steps)
  hca_past = shift_forward(signals['hca_curv_signed'], shift_steps)
  qfk_past = shift_forward(signals['qfk_curv_signed'], shift_steps)
  actuator_past = shift_forward(signals['cc_actuator_curv'], shift_steps)

  qfk_now = signals['qfk_curv_signed']
  hca_now = signals['hca_curv_signed']

  v = signals['v_ego']
  idxs = np.where(learn_mask)[0]
  for i in idxs:
    desired = model_raw_past[i]
    s_idx = speed_index(float(v[i]))
    c_idx = curvature_index(float(desired))
    if s_idx is None or c_idx is None:
      continue
    sgn_idx = sign_index(float(desired))

    # R_model_yaw: model said go to desired at t-lag; vehicle is at yaw_curv now
    err_my = float(desired - yaw_curv[i])
    acc.add('R_model_yaw', s_idx, c_idx, sgn_idx, err_my)

    if require_qfk:
      err_mq = float(desired - qfk_now[i])
      acc.add('R_model_qfk', s_idx, c_idx, sgn_idx, err_mq)

    # R_smooth_loss: model raw vs controller-output actuator curvature at the
    # same instant (no lag shift needed - both at t-lag)
    err_sl = float(desired - actuator_past[i])
    acc.add('R_smooth_loss', s_idx, c_idx, sgn_idx, err_sl)

    # R_hca_qfk: HCA commanded at t-lag vs QFK measured now (EPS execution)
    if require_hca and require_qfk:
      err_hq = float(hca_past[i] - qfk_now[i])
      # bucket by |hca_past| sign + magnitude
      h_c = curvature_index(float(hca_past[i]))
      if h_c is not None:
        h_s = sign_index(float(hca_past[i]))
        acc.add('R_hca_qfk', s_idx, h_c, h_s, err_hq)

    # R_hca_yaw: HCA commanded at t-lag vs yaw actual now
    if require_hca:
      err_hy = float(hca_past[i] - yaw_curv[i])
      h_c = curvature_index(float(hca_past[i]))
      if h_c is not None:
        h_s = sign_index(float(hca_past[i]))
        acc.add('R_hca_yaw', s_idx, h_c, h_s, err_hy)

  return acc


# ---------- plant identification (ARX 1,1,d) ----------

# Plant fit acceptance criteria. Anything outside these bounds is rejected
# rather than reported - matches the earlier-cited failure mode of K>1 results
# (impossible for a passive plant) being treated as data.
PLANT_K_BOUNDS = (0.5, 1.5)
PLANT_T_BOUNDS_S = (0.02, 1.0)
PLANT_A_BOUNDS = (0.05, 0.99)
PLANT_MIN_R2 = 0.5
PLANT_MAX_DELAY_STEPS = 6   # at DT_GRID=0.05s -> 0..300ms pure-delay search

# Speed bins for per-band plant aggregation
PLANT_SPEED_BINS_KMH = (20, 40, 60, 80, 100, 120, 140)
PLANT_MIN_RUN_S = 5.0       # contiguous engagement needed for one fit


def _find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
  """Return [(start, end)] inclusive index pairs for contiguous True runs >= min_len."""
  if mask.sum() == 0:
    return []
  edges = np.diff(mask.astype(np.int8))
  starts = np.where(edges == 1)[0] + 1
  ends = np.where(edges == -1)[0]
  if mask[0]:
    starts = np.concatenate(([0], starts))
  if mask[-1]:
    ends = np.concatenate((ends, [len(mask) - 1]))
  return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]


def fit_arx_1_1_d(u: np.ndarray, y: np.ndarray, dt: float = DT_GRID,
                  max_d: int = PLANT_MAX_DELAY_STEPS) -> dict | None:
  """First-order ARX with pure delay: y[n] = a*y[n-1] + b*u[n-d].

  Returns the best (a, b, d) by SSR among d in [0, max_d] that passes physical
  bounds. None if no candidate qualifies.

  Caveat (recorded explicitly): u and y are from closed-loop operation, so the
  fitted (K, T) is the effective plant as seen by the controller, not an
  open-loop plant. For feed-forward inversion that is the right thing to fit.
  """
  if u is None or y is None or len(u) != len(y):
    return None
  u = np.asarray(u, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  if u.size < max_d + 40:
    return None
  # DC-remove so the fit isn't dominated by a bias term
  u = u - u.mean()
  y = y - y.mean()
  if np.std(u) < 1e-7 or np.std(y) < 1e-7:
    return None
  N = u.size
  best: dict | None = None
  y_target = y[max_d + 1 :]
  y_lag = y[max_d:-1]
  for d in range(0, max_d + 1):
    u_lag = u[max_d + 1 - d : N - d]
    X = np.column_stack([y_lag, u_lag])
    try:
      coef, *_ = np.linalg.lstsq(X, y_target, rcond=None)
    except np.linalg.LinAlgError:
      continue
    a, b = float(coef[0]), float(coef[1])
    if not (PLANT_A_BOUNDS[0] < a < PLANT_A_BOUNDS[1]):
      continue
    if b <= 0:
      continue
    K = b / (1.0 - a)
    if not (PLANT_K_BOUNDS[0] < K < PLANT_K_BOUNDS[1]):
      continue
    T = -dt / math.log(a)
    if not (PLANT_T_BOUNDS_S[0] < T < PLANT_T_BOUNDS_S[1]):
      continue
    y_pred = X @ coef
    ssr = float(np.sum((y_target - y_pred) ** 2))
    sst = float(np.sum(y_target ** 2)) + 1e-12
    r2 = 1.0 - ssr / sst
    if r2 < PLANT_MIN_R2:
      continue
    cand = {
      'K': K, 'T_s': T, 'tau_s': float(d * dt),
      'a': a, 'b': b, 'R2': r2, 'ssr': ssr,
      'n': int(y_target.size), 'd_best': d,
    }
    if best is None or ssr < best['ssr']:
      best = cand
  return best


def assign_speed_bin(v_kmh: float) -> int | None:
  """Bin centers: 20, 40, 60, 80, 100, 120, 140 km/h with +/-10 width each."""
  for b in PLANT_SPEED_BINS_KMH:
    if abs(v_kmh - b) <= 10:
      return b
  return None


def plant_id_per_segment(signals: dict, apply_mask: np.ndarray) -> list[dict]:
  """Find contiguous engaged runs and fit a first-order plant on each.

  Two plant interpretations are fit separately:
    'eps_fit'   : HCA_03 cmd -> QFK_01 measured   (EPS inner loop)
    'plant_fit' : HCA_03 cmd -> yaw-rate-derived  (full EPS + vehicle response)
  """
  if signals['hca_curv_signed'].size == 0 or signals['qfk_curv_signed'].size == 0:
    return []
  min_len = int(PLANT_MIN_RUN_S * RESAMPLE_HZ)
  runs = _find_runs(apply_mask, min_len)
  v_kmh = signals['v_ego'] * 3.6
  yaw_curv = compute_yaw_actual_curvature(signals)
  u = signals['hca_curv_signed']
  y_qfk = signals['qfk_curv_signed']

  fits: list[dict] = []
  for s, e in runs:
    sl = slice(s, e + 1)
    v_mean = float(v_kmh[sl].mean())
    v_span = float(v_kmh[sl].max() - v_kmh[sl].min())
    bin_kmh = assign_speed_bin(v_mean)
    if bin_kmh is None:
      continue
    fits.append({
      'start_idx': s, 'end_idx': e, 'n_samples': int(e - s + 1),
      'v_mean_kmh': v_mean, 'v_span_kmh': v_span, 'speed_bin_kmh': bin_kmh,
      'eps_fit': fit_arx_1_1_d(u[sl], y_qfk[sl]),
      'plant_fit': fit_arx_1_1_d(u[sl], yaw_curv[sl]),
    })
  return fits


# ---------- lag estimation ----------

def estimate_lags(signals: dict, learn_mask: np.ndarray) -> dict:
  """Cross-correlate d/dt(model_raw_desired) vs d/dt(yaw_curv) in three
  speed bands. Returns argmax lag in seconds for each band."""
  out: dict[str, dict] = {}
  if learn_mask.sum() == 0 or signals['model_raw_desired'].size == 0:
    return {f'{a}_{b}kmh': {'lag_s': None, 'samples': 0} for a, b in SPEED_BANDS_KMH}

  yaw_curv = compute_yaw_actual_curvature(signals)
  d_desired = np.gradient(signals['model_raw_desired'], DT_GRID)
  d_actual = np.gradient(yaw_curv, DT_GRID)
  v_kmh = signals['v_ego'] * 3.6
  max_shift = int(round(LAG_SEARCH_S / DT_GRID))
  shifts = np.arange(-max_shift, max_shift + 1)

  for lo, hi in SPEED_BANDS_KMH:
    band = learn_mask & (v_kmh >= lo) & (v_kmh < hi)
    n = int(band.sum())
    band_key = f'{lo}_{hi}kmh'
    if n < max_shift * 4:
      out[band_key] = {'lag_s': None, 'samples': n}
      continue
    a = d_desired - d_desired[band].mean()
    b = d_actual - d_actual[band].mean()
    a[~band] = 0.0
    b[~band] = 0.0
    xcorr = np.array([float(np.sum(a[max(0, s):len(a) + min(0, s)] * b[max(0, -s):len(b) + min(0, -s)]))
                      for s in shifts])
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    xcorr /= norm
    best = int(np.argmax(xcorr))
    lag_s = float(shifts[best]) * DT_GRID
    out[band_key] = {
      'lag_s': lag_s,
      'samples': n,
      'peak_xcorr': float(xcorr[best]),
      'zero_xcorr': float(xcorr[max_shift]),
    }
  return out


# ---------- deadband and asymmetry ----------

def deadband_signature(acc: ResidualAccum) -> dict:
  """Signed bias of R_model_yaw in the smallest 4 curvature buckets, separated
  by sign of desired curvature."""
  buckets = slice(0, 4)
  out = {}
  for sgn in (0, 1):
    c = acc.count['R_model_yaw'][:, buckets, sgn].sum()
    s = acc.sum['R_model_yaw'][:, buckets, sgn].sum()
    out[f'sign_{sgn}'] = {
      'count': int(c),
      'mean_signed_bias': float(s / c) if c > 0 else None,
    }
  return out


def asymmetry_signature(acc: ResidualAccum) -> dict:
  """Difference of mean R_model_yaw between positive and negative desired
  curvature, per speed."""
  out = {}
  for s_idx, kmh in enumerate(SPEED_ANCHORS_KMH):
    pos_c = acc.count['R_model_yaw'][s_idx, :, 1].sum()
    neg_c = acc.count['R_model_yaw'][s_idx, :, 0].sum()
    pos_s = acc.sum['R_model_yaw'][s_idx, :, 1].sum()
    neg_s = acc.sum['R_model_yaw'][s_idx, :, 0].sum()
    pos_mean = float(pos_s / pos_c) if pos_c > 0 else None
    neg_mean = float(neg_s / neg_c) if neg_c > 0 else None
    diff = (pos_mean + neg_mean) if (pos_mean is not None and neg_mean is not None) else None
    # NOTE: bias is sign-projected via projected_error in the learner; here
    # we keep raw signed (desired - actual). For symmetric plant we'd
    # expect pos_mean and neg_mean to have opposite signs and equal mag;
    # their *sum* should be near zero if symmetric.
    out[f'{int(kmh)}kmh'] = {
      'pos_mean': pos_mean,
      'neg_mean': neg_mean,
      'sum': diff,  # asymmetry indicator; non-zero -> left/right mismatch
      'pos_count': int(pos_c),
      'neg_count': int(neg_c),
    }
  return out


# ---------- metadata ----------

def _detect_branch(raw: RouteRaw) -> str:
  if raw.has_liveCurvatureParameters:
    return "sunnypilot_learner"
  if raw.has_modelDesiredCurvature:
    return "sunnypilot_pre_learner"
  # Could refine by inspecting carParams.openpilotLongitudinalControl,
  # carParams.flags etc. For now, comma is the default fallback.
  return "comma_or_unknown"


def summarise_metadata(route_id: str, raw: RouteRaw, signals: dict,
                       apply_mask: np.ndarray, learn_mask: np.ndarray,
                       lat_delay_s: float) -> dict:
  cp = raw.car_params
  meta: dict[str, Any] = {
    'route_id': route_id,
    'version': VERSION,
    'fingerprint': str(getattr(cp, 'carFingerprint', '')) if cp else '',
    'brand': str(getattr(cp, 'brand', '')) if cp else '',
    'vin': str(getattr(cp, 'carVin', '')) if cp else '',
    'fw_versions': len(getattr(cp, 'carFw', []) or []) if cp else 0,
    'flags': int(getattr(cp, 'flags', 0)) if cp else 0,
    'steer_control_type': str(getattr(cp, 'steerControlType', '')) if cp else '',
    'branch_inferred': _detect_branch(raw),
    'has_can': raw.has_can,
    'can_bus_used': raw.can_bus_used,
    'has_modelDesiredCurvature_field': raw.has_modelDesiredCurvature,
    'has_liveCurvatureParameters': raw.has_liveCurvatureParameters,
    'log_counts': dict(raw.log_count_by_type),
    'grid_samples': int(signals['v_ego'].size) if signals else 0,
    'apply_samples': int(apply_mask.sum()),
    'learn_samples': int(learn_mask.sum()),
    'duration_s': float((signals['v_ego'].size * DT_GRID) if signals else 0.0),
    'lat_delay_used_s': lat_delay_s,
  }

  # liveParameters / liveDelay summary (means over learn span)
  if signals:
    for k in ('lp_steer_ratio', 'lp_stiffness', 'lp_angle_offset', 'lp_roll',
             'ld_lat_delay', 'ld_lat_delay_est'):
      vals = signals[k][learn_mask] if learn_mask.size else np.zeros(0)
      meta[k] = {
        'mean': float(np.mean(vals)) if vals.size else None,
        'std': float(np.std(vals)) if vals.size else None,
        'n': int(vals.size),
      }
    # speed distribution
    v_kmh = signals['v_ego'] * 3.6
    meta['speed_kmh'] = {
      'p10': float(np.percentile(v_kmh[apply_mask], 10)) if apply_mask.sum() else None,
      'p50': float(np.percentile(v_kmh[apply_mask], 50)) if apply_mask.sum() else None,
      'p90': float(np.percentile(v_kmh[apply_mask], 90)) if apply_mask.sum() else None,
      'p99': float(np.percentile(v_kmh[apply_mask], 99)) if apply_mask.sum() else None,
    }
    # driver torque / steering activity
    if signals['eps_torque'].size:
      meta['eps_torque_abs_mean'] = float(np.mean(np.abs(signals['eps_torque'][apply_mask])))
    # saturations on the lateral controller side: hca status != active
    if signals['hca_request_status'].size:
      active = signals['hca_request_status'] == 4
      meta['hca_active_frac'] = float(np.mean(active.astype(np.float32)))
  return meta


# ---------- top-level ----------

def extract_route(route_id: str, out_dir: str | None = None,
                  max_seconds: float | None = None,
                  output_name: str | None = None) -> dict:
  t0 = time.time()
  raw = collect_route(route_id, max_seconds=max_seconds)
  t_grid, signals = build_grid(raw)
  if t_grid.size == 0:
    summary = {
      'route_id': route_id,
      'version': VERSION,
      'error': 'no_overlap',
      'log_counts': dict(raw.log_count_by_type),
      'has_can': raw.has_can,
    }
  else:
    apply_mask, learn_mask = build_engagement_mask(t_grid, signals)
    lat_delay_s = median_lat_delay(signals, learn_mask if learn_mask.sum() else apply_mask)
    require_hca = signals['hca_curv_signed'].any()
    require_qfk = signals['qfk_curv_signed'].any()
    acc = accumulate_residuals(signals, learn_mask, lat_delay_s, require_hca, require_qfk)
    lags = estimate_lags(signals, learn_mask)
    deadband = deadband_signature(acc)
    asymmetry = asymmetry_signature(acc)
    plant_fits = plant_id_per_segment(signals, apply_mask) if (require_hca and require_qfk) else []
    meta = summarise_metadata(route_id, raw, signals, apply_mask, learn_mask, lat_delay_s)
    summary = {
      **meta,
      'residuals': {
        name: {
          'count': acc.count[name],
          'sum': acc.sum[name],
          'sum_sq': acc.sum_sq[name],
          'sum_abs': acc.sum_abs[name],
        } for name in RESIDUAL_NAMES
      },
      'lags': lags,
      'deadband': deadband,
      'asymmetry': asymmetry,
      'plant_fits': plant_fits,
      'bucket_grid': {
        'speed_anchors_kmh': SPEED_ANCHORS_KMH.tolist(),
        'curv_edges': CURV_EDGES.tolist(),
        'curv_centers': CURV_CENTERS.tolist(),
      },
    }

  summary['extract_walltime_s'] = float(time.time() - t0)

  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    safe_src = output_name if output_name is not None else route_id
    safe = safe_src.replace('/', '__').replace('|', '__').replace(':', '_')
    path = os.path.join(out_dir, f'{safe}.pkl')
    with open(path, 'wb') as f:
      pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary['_pickle_path'] = path

  return summary


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('route_id', help='dongle/route[/segment_or_a]')
  ap.add_argument('--out-dir', default='tools/lateral_maneuvers/fleet_residuals/out')
  ap.add_argument('--max-seconds', type=float, default=None,
                  help='Process at most N seconds of route (for fast iteration).')
  ap.add_argument('--quiet', action='store_true')
  args = ap.parse_args()

  try:
    s = extract_route(args.route_id, out_dir=args.out_dir, max_seconds=args.max_seconds)
  except Exception:
    print(f'FAIL {args.route_id}', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

  if args.quiet:
    return

  if 'error' in s:
    print(f"{args.route_id}: error={s['error']} log_counts={s.get('log_counts')}")
    return

  print(f"route_id          : {s['route_id']}")
  print(f"fingerprint       : {s['fingerprint']}")
  print(f"brand             : {s['brand']}")
  print(f"branch_inferred   : {s['branch_inferred']}")
  print(f"has_can           : {s['has_can']} (hca_bus,qfk_bus={s['can_bus_used']})")
  print(f"duration_s        : {s['duration_s']:.1f}")
  print(f"apply_samples     : {s['apply_samples']}")
  print(f"learn_samples     : {s['learn_samples']}")
  print(f"lat_delay_used_s  : {s['lat_delay_used_s']:.3f}")
  if s.get('speed_kmh'):
    print(f"speed p10/50/90/99: {s['speed_kmh']['p10']}/{s['speed_kmh']['p50']}/{s['speed_kmh']['p90']}/{s['speed_kmh']['p99']}")
  print(f"lp_steer_ratio    : mean={s['lp_steer_ratio']['mean']} std={s['lp_steer_ratio']['std']}")
  print(f"ld_lat_delay      : mean={s['ld_lat_delay']['mean']} std={s['ld_lat_delay']['std']}")

  for name in RESIDUAL_NAMES:
    r = s['residuals'][name]
    c = r['count']
    if c.sum() == 0:
      print(f"  {name}: no samples")
      continue
    total = int(c.sum())
    sgn = r['sum']
    mean = float(sgn.sum() / total)
    rms = float(np.sqrt(r['sum_sq'].sum() / total))
    mae = float(r['sum_abs'].sum() / total)
    print(f"  {name}: n={total} mean={mean:+.2e} rms={rms:.2e} mae={mae:.2e}")

  for band, info in s['lags'].items():
    if info['lag_s'] is None:
      print(f"  lag {band}: insufficient samples (n={info['samples']})")
    else:
      print(f"  lag {band}: {info['lag_s']:+.3f} s (n={info['samples']}, peak={info.get('peak_xcorr', 0):.2f})")

  if s.get('_pickle_path'):
    print(f"pickle            : {s['_pickle_path']}")


if __name__ == '__main__':
  main()
