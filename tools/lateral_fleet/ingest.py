"""Per-route / per-segment extraction.

Two entry points:

* `process_route(dongle, route)` — full route via canonical URI; LogReader
  resolves all segments via the comma data sources.
* `process_url(dongle, segment_id, rlog_url)` — single segment via direct
  URL. Used by the eps_seglist.csv ingest path.

In both cases the output is one parquet timeline + one parquet bucket-stats
file per route/segment, plus a JSON status sidecar. No script holds more
than one route's full timeline in memory.
"""
from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from opendbc.can import CANParser

from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader, LogsUnavailable, ReadMode, _LogFileReader

from openpilot.tools.lateral_fleet import cache, features

log = logging.getLogger(__name__)

DBC_NAME = 'vw_meb'
CANDIDATE_BUSES = (0, 2, 128, 130)

HCA_03 = 'HCA_03'
QFK_01 = 'QFK_01'
LH_EPS_03 = 'LH_EPS_03'

CAN_MESSAGES = [(HCA_03, 50), (QFK_01, 100), (LH_EPS_03, 100)]

HCA_03_ADDR = 771
QFK_01_ADDR = 317
LH_EPS_03_ADDR = 159

MSG_ADDRS = {HCA_03: HCA_03_ADDR, QFK_01: QFK_01_ADDR, LH_EPS_03: LH_EPS_03_ADDR}

TARGET_FINGERPRINTS = {'VOLKSWAGEN_ID4_MK1'}

_CAL_VALUES = {'uncalibrated': 0, 'calibrated': 1, 'invalid': 2, 'recalibrating': 3}


@dataclass
class _HcaSnap:
  hca_curv: float
  hca_vz: int
  hca_request_status: int
  hca_power: float


@dataclass
class _QfkSnap:
  qfk_curv: float
  qfk_vz: int
  qfk_hca_status: int


@dataclass
class _EpsSnap:
  eps_lenkmoment: float
  eps_vz: int
  eps_hca_status: int


def _snap_hca(cp: CANParser) -> _HcaSnap:
  v = cp.vl[HCA_03]
  return _HcaSnap(float(v['Curvature']), int(v['Curvature_VZ']),
                  int(v['RequestStatus']), float(v['Power']))


def _snap_qfk(cp: CANParser) -> _QfkSnap:
  v = cp.vl[QFK_01]
  return _QfkSnap(float(v['Curvature']), int(v['Curvature_VZ']),
                  int(v['LatCon_HCA_Status']))


def _snap_eps(cp: CANParser) -> _EpsSnap:
  v = cp.vl[LH_EPS_03]
  return _EpsSnap(float(v['EPS_Lenkmoment']), int(v['EPS_VZ_Lenkmoment']),
                  int(v['EPS_HCA_Status']))


def _route_uri(dongle_id: str, route_id: str) -> str:
  return f'{dongle_id}|{route_id}/a'


_SEG_RE = re.compile(r'/(\d+)/[rq]log\.')


def _seg_num(uri: str) -> int:
  m = _SEG_RE.search(uri)
  return int(m.group(1)) if m else -1


def _iter_segments_in_order(lr: LogReader):
  """LogReader assembles files from multiple sources, so logreader_identifiers
  may be out of segment order. Mono-time across segments is continuous on a
  single drive, so we just need to consume segments in segment order."""
  uris = sorted(lr.logreader_identifiers, key=_seg_num)
  for u in uris:
    yield from _LogFileReader(u, sort_by_time=True)


def _iter_single(rlog_url: str):
  yield from _LogFileReader(rlog_url, sort_by_time=True)


# ---------------------------------------------------------------------------
# Cereal extraction
# ---------------------------------------------------------------------------

def _extract_carstate(cs) -> dict:
  return {
    'v_ego': float(cs.vEgo),
    'yaw_rate': float(cs.yawRate),
    'steering_angle_deg': float(cs.steeringAngleDeg),
    'steering_pressed': bool(cs.steeringPressed),
  }


def _extract_carcontrol(cc) -> dict:
  return {
    'lat_active': bool(cc.latActive),
    'c_desired': float(cc.actuators.curvature),
    'c_vm': float(cc.currentCurvature),
  }


def _extract_controlsstate(c) -> dict:
  return {'c_desiredCS': float(c.desiredCurvature)}


def _extract_liveparams(lp) -> dict:
  return {
    'steer_ratio': float(lp.steerRatio),
    'stiffness_factor': float(lp.stiffnessFactor),
    'roll': float(lp.roll),
    'angle_offset_deg': float(lp.angleOffsetDeg),
    'lp_valid': bool(lp.valid),
  }


def _extract_livedelay(ld) -> dict:
  return {'lateral_delay': float(ld.lateralDelay)}


def _extract_livepose(lp) -> dict:
  ang = lp.angularVelocityDevice
  return {
    'pose_yaw_rate': float(ang.z),
    'pose_yaw_rate_std': float(getattr(ang, 'zStd', 0.0) or 0.0),
    'pose_inputs_ok': bool(lp.inputsOK),
    'pose_posenet_ok': bool(lp.posenetOK),
  }


def _extract_calib(cal) -> dict:
  status_raw = cal.calStatus
  try:
    status_val = int(status_raw)
  except (TypeError, ValueError):
    status_val = _CAL_VALUES.get(str(status_raw), 0)
  return {'calib_status': status_val}


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _forward_fill_resample(events: list[tuple[int, dict]], grid_ns: np.ndarray,
                           cols: list[str]) -> dict[str, np.ndarray]:
  out = {c: np.full(grid_ns.size, np.nan, dtype=np.float64) for c in cols}
  if not events:
    return out
  events = sorted(events, key=lambda e: e[0])
  ts = np.array([e[0] for e in events], dtype=np.int64)
  idx = np.searchsorted(ts, grid_ns, side='right') - 1
  for c in cols:
    vals = np.array([float(e[1].get(c, np.nan)) for e in events], dtype=np.float64)
    fill = np.full(grid_ns.size, np.nan)
    in_range = idx >= 0
    fill[in_range] = vals[idx[in_range]]
    out[c] = fill
  return out


def _resample_snap_stream(snaps: list[tuple[int, object]], grid_ns: np.ndarray,
                          field_names: list[str]) -> dict[str, np.ndarray]:
  out = {c: np.full(grid_ns.size, np.nan) for c in field_names}
  if not snaps:
    return out
  snaps = sorted(snaps, key=lambda s: s[0])
  ts = np.array([s[0] for s in snaps], dtype=np.int64)
  idx = np.searchsorted(ts, grid_ns, side='right') - 1
  in_range = idx >= 0
  for c in field_names:
    vals = np.array([getattr(s[1], c) for s in snaps], dtype=np.float64)
    fill = np.full(grid_ns.size, np.nan)
    fill[in_range] = vals[idx[in_range]]
    out[c] = fill
  return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def process_route(dongle_id: str, route_id: str) -> cache.RouteStatus:
  if cache.already_ok(dongle_id, route_id):
    return cache.read_status(dongle_id, route_id)  # type: ignore[return-value]
  try:
    lr = LogReader(_route_uri(dongle_id, route_id), default_mode=ReadMode.AUTO)
    return _run(dongle_id, route_id, _iter_segments_in_order(lr))
  except LogsUnavailable as e:
    rs = cache.RouteStatus(dongle_id, route_id, 'missing', str(e))
    cache.write_status(rs)
    return rs
  except Exception as e:  # noqa: BLE001
    log.exception('process_route failed for %s/%s', dongle_id, route_id)
    rs = cache.RouteStatus(dongle_id, route_id, 'corrupted',
                           f'{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}')
    cache.write_status(rs)
    return rs


def process_url(dongle_id: str, segment_id: str, rlog_url: str) -> cache.RouteStatus:
  if cache.already_ok(dongle_id, segment_id):
    return cache.read_status(dongle_id, segment_id)  # type: ignore[return-value]
  try:
    return _run(dongle_id, segment_id, _iter_single(rlog_url))
  except LogsUnavailable as e:
    rs = cache.RouteStatus(dongle_id, segment_id, 'missing', str(e))
    cache.write_status(rs)
    return rs
  except Exception as e:  # noqa: BLE001
    log.exception('process_url failed for %s/%s', dongle_id, segment_id)
    rs = cache.RouteStatus(dongle_id, segment_id, 'corrupted',
                           f'{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}')
    cache.write_status(rs)
    return rs


# ---------------------------------------------------------------------------
# Single-pass extraction
# ---------------------------------------------------------------------------

def _run(dongle_id: str, route_id: str, msg_iter) -> cache.RouteStatus:
  cs_events: list[tuple[int, dict]] = []
  cc_events: list[tuple[int, dict]] = []
  ctl_events: list[tuple[int, dict]] = []
  lp_events: list[tuple[int, dict]] = []
  ld_events: list[tuple[int, dict]] = []
  pose_events: list[tuple[int, dict]] = []
  cal_events: list[tuple[int, dict]] = []
  carparams = None
  has_lcp = False

  parsers = {b: CANParser(DBC_NAME, CAN_MESSAGES, b) for b in CANDIDATE_BUSES}
  hca_snaps: dict[int, list[tuple[int, _HcaSnap]]] = {b: [] for b in CANDIDATE_BUSES}
  qfk_snaps: dict[int, list[tuple[int, _QfkSnap]]] = {b: [] for b in CANDIDATE_BUSES}
  eps_snaps: dict[int, list[tuple[int, _EpsSnap]]] = {b: [] for b in CANDIDATE_BUSES}
  bus_counts: dict[str, dict[int, int]] = {
    HCA_03: {b: 0 for b in CANDIDATE_BUSES},
    QFK_01: {b: 0 for b in CANDIDATE_BUSES},
    LH_EPS_03: {b: 0 for b in CANDIDATE_BUSES},
  }

  for msg in msg_iter:
    try:
      w = msg.which()
    except Exception:
      continue
    if w == 'carState':
      cs_events.append((int(msg.logMonoTime), _extract_carstate(msg.carState)))
    elif w == 'carControl':
      cc_events.append((int(msg.logMonoTime), _extract_carcontrol(msg.carControl)))
    elif w == 'controlsState':
      ctl_events.append((int(msg.logMonoTime), _extract_controlsstate(msg.controlsState)))
    elif w == 'liveParameters':
      lp_events.append((int(msg.logMonoTime), _extract_liveparams(msg.liveParameters)))
    elif w == 'liveDelay':
      ld_events.append((int(msg.logMonoTime), _extract_livedelay(msg.liveDelay)))
    elif w == 'livePose':
      pose_events.append((int(msg.logMonoTime), _extract_livepose(msg.livePose)))
    elif w == 'liveCalibration':
      cal_events.append((int(msg.logMonoTime), _extract_calib(msg.liveCalibration)))
    elif w == 'carParams' and carparams is None:
      carparams = {
        'fingerprint': str(msg.carParams.carFingerprint),
        'brand': str(msg.carParams.brand),
        'vin': str(msg.carParams.carVin) if hasattr(msg.carParams, 'carVin') else '',
      }
    elif w == 'liveCurvatureParameters':
      has_lcp = True
    elif w == 'can':
      try:
        packets = can_capnp_to_list([msg.as_builder().to_bytes()])
      except Exception:
        continue
      tally = {m: {b: 0 for b in CANDIDATE_BUSES} for m in MSG_ADDRS}
      for _ns, frames in packets:
        for addr, _dat, src in frames:
          if src not in CANDIDATE_BUSES:
            continue
          for mn, ma in MSG_ADDRS.items():
            if addr == ma:
              tally[mn][src] += 1
              bus_counts[mn][src] += 1
      mono = int(msg.logMonoTime)
      for b, cp in parsers.items():
        cp.update(packets)
        if tally[HCA_03][b]:
          hca_snaps[b].append((mono, _snap_hca(cp)))
        if tally[QFK_01][b]:
          qfk_snaps[b].append((mono, _snap_qfk(cp)))
        if tally[LH_EPS_03][b]:
          eps_snaps[b].append((mono, _snap_eps(cp)))

  if carparams is None:
    rs = cache.RouteStatus(dongle_id, route_id, 'wrong_fingerprint', 'no carParams in log')
    cache.write_status(rs)
    return rs
  if carparams['fingerprint'] not in TARGET_FINGERPRINTS:
    rs = cache.RouteStatus(dongle_id, route_id, 'wrong_fingerprint',
                           f"fingerprint={carparams['fingerprint']}")
    cache.write_status(rs)
    return rs

  if all(c == 0 for c in bus_counts[HCA_03].values()):
    rs = cache.RouteStatus(dongle_id, route_id, 'no_can_bus',
                           f'no HCA_03 frames on any of {CANDIDATE_BUSES}; '
                           f'qlog-only-missing-can probable')
    cache.write_status(rs)
    return rs

  chosen_buses = {
    HCA_03: max(bus_counts[HCA_03], key=bus_counts[HCA_03].get),
    QFK_01: max(bus_counts[QFK_01], key=bus_counts[QFK_01].get),
    LH_EPS_03: max(bus_counts[LH_EPS_03], key=bus_counts[LH_EPS_03].get),
  }
  chosen_hca_snaps = hca_snaps[chosen_buses[HCA_03]]
  chosen_qfk_snaps = qfk_snaps[chosen_buses[QFK_01]]
  chosen_eps_snaps = eps_snaps[chosen_buses[LH_EPS_03]]
  hca_snaps.clear(); qfk_snaps.clear(); eps_snaps.clear()

  starts = [cs_events[0][0] if cs_events else None,
            chosen_hca_snaps[0][0] if chosen_hca_snaps else None,
            chosen_qfk_snaps[0][0] if chosen_qfk_snaps else None,
            chosen_eps_snaps[0][0] if chosen_eps_snaps else None]
  ends = [cs_events[-1][0] if cs_events else None,
          chosen_hca_snaps[-1][0] if chosen_hca_snaps else None,
          chosen_qfk_snaps[-1][0] if chosen_qfk_snaps else None,
          chosen_eps_snaps[-1][0] if chosen_eps_snaps else None]
  start_ns = min(s for s in starts if s is not None) if any(s is not None for s in starts) else 0
  end_ns = max(e for e in ends if e is not None) if any(e is not None for e in ends) else 0
  if end_ns <= start_ns:
    rs = cache.RouteStatus(dongle_id, route_id, 'corrupted', 'empty time range')
    cache.write_status(rs)
    return rs

  step_ns = int(round(1e9 / features.RESAMPLE_HZ))
  grid_ns = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)

  cs_grid = _forward_fill_resample(cs_events, grid_ns,
                                   ['v_ego', 'yaw_rate', 'steering_angle_deg', 'steering_pressed'])
  cc_grid = _forward_fill_resample(cc_events, grid_ns, ['lat_active', 'c_desired', 'c_vm'])
  ctl_grid = _forward_fill_resample(ctl_events, grid_ns, ['c_desiredCS'])
  lp_grid = _forward_fill_resample(lp_events, grid_ns,
                                   ['steer_ratio', 'stiffness_factor', 'roll',
                                    'angle_offset_deg', 'lp_valid'])
  ld_grid = _forward_fill_resample(ld_events, grid_ns, ['lateral_delay'])
  pose_grid = _forward_fill_resample(pose_events, grid_ns,
                                     ['pose_yaw_rate', 'pose_yaw_rate_std',
                                      'pose_inputs_ok', 'pose_posenet_ok'])
  cal_grid = _forward_fill_resample(cal_events, grid_ns, ['calib_status'])
  hca_grid = _resample_snap_stream(chosen_hca_snaps, grid_ns,
                                   ['hca_curv', 'hca_vz', 'hca_request_status', 'hca_power'])
  qfk_grid = _resample_snap_stream(chosen_qfk_snaps, grid_ns,
                                   ['qfk_curv', 'qfk_vz', 'qfk_hca_status'])
  eps_grid = _resample_snap_stream(chosen_eps_snaps, grid_ns,
                                   ['eps_lenkmoment', 'eps_vz', 'eps_hca_status'])

  timeline = pd.DataFrame({
    'mono_ns': grid_ns,
    **cs_grid, **cc_grid, **ctl_grid, **lp_grid, **ld_grid, **pose_grid, **cal_grid,
    **hca_grid, **qfk_grid, **eps_grid,
  })

  required = ['v_ego', 'lat_active', 'c_desired', 'c_vm',
              'steer_ratio', 'roll', 'hca_curv', 'qfk_curv']
  ok_mask = ~timeline[required].isna().any(axis=1)
  if not ok_mask.any():
    rs = cache.RouteStatus(dongle_id, route_id, 'corrupted', 'no overlapping signals')
    cache.write_status(rs)
    return rs
  timeline = timeline.loc[ok_mask].reset_index(drop=True)

  hca_sign = np.where(timeline['hca_vz'].to_numpy() == 1, 1.0, -1.0)
  qfk_sign = np.where(timeline['qfk_vz'].to_numpy() == 1, 1.0, -1.0)
  eps_sign = np.where(timeline['eps_vz'].to_numpy() == 0, 1.0, -1.0)
  timeline['c_cmd'] = timeline['hca_curv'] * hca_sign
  timeline['c_eps'] = timeline['qfk_curv'] * qfk_sign
  timeline['torque_driver'] = timeline['eps_lenkmoment'] * eps_sign / 100.0
  timeline['hca_power_pct'] = timeline['hca_power'] * 0.4
  timeline['hca_power'] = timeline['hca_power_pct']

  yaw = timeline['pose_yaw_rate'].to_numpy()
  fallback = timeline['yaw_rate'].to_numpy()
  yaw = np.where(np.isnan(yaw), fallback, yaw)
  timeline['c_yaw'] = features.curvature_from_yaw(yaw, timeline['v_ego'].to_numpy())

  delay = float(np.nanmedian(timeline['lateral_delay'].to_numpy()))
  if not np.isfinite(delay) or delay <= 0:
    delay = 0.15
  timeline['c_cmd_delayed'] = features.shift_by_delay(timeline['c_cmd'].to_numpy(), delay)

  timeline['resid_yaw'] = timeline['c_yaw'].to_numpy() - timeline['c_cmd_delayed'].to_numpy()
  timeline['resid_eps'] = timeline['c_eps'].to_numpy() - timeline['c_cmd_delayed'].to_numpy()
  timeline['gain_yaw'] = features.compute_gain_ratio(
    timeline['c_yaw'].to_numpy(), timeline['c_cmd_delayed'].to_numpy())
  timeline['gain_eps'] = features.compute_gain_ratio(
    timeline['c_eps'].to_numpy(), timeline['c_cmd_delayed'].to_numpy())

  lat_active = timeline['lat_active'].fillna(0).astype(bool).to_numpy()
  steering_pressed = timeline['steering_pressed'].fillna(0).astype(bool).to_numpy()
  v_ego = timeline['v_ego'].to_numpy()
  yaw_rate_std = timeline['pose_yaw_rate_std'].fillna(0).to_numpy()
  roll = timeline['roll'].to_numpy()
  calib_status = timeline['calib_status'].fillna(0).astype(int).to_numpy()
  c_cmd = timeline['c_cmd'].to_numpy()

  mask_loose = features.loose_gates(lat_active, steering_pressed, v_ego)
  mask_strict = features.strict_gates(
    v_ego=v_ego, yaw_rate=yaw, yaw_rate_std=yaw_rate_std, roll_rad=roll,
    lat_active=lat_active, steering_pressed=steering_pressed,
    calib_status=calib_status, c_cmd=c_cmd,
  )

  timeline['mask_loose'] = mask_loose
  timeline['mask_strict'] = mask_strict

  if int(mask_strict.sum()) == 0:
    rs = cache.RouteStatus(
      dongle_id, route_id, 'no_engaged_time',
      f'no samples pass strict gates (loose={int(mask_loose.sum())})',
      fingerprint=carparams['fingerprint'], vin=carparams['vin'],
      can_bus=int(chosen_buses[HCA_03]))
    cache.write_status(rs)
    return rs

  buckets = features.bucket_route(timeline, mask_strict)
  buckets['dongle_id'] = dongle_id
  buckets['route_id'] = route_id
  buckets['fingerprint'] = carparams['fingerprint']
  buckets['vin'] = carparams['vin']
  buckets['lcp_seen'] = has_lcp

  cache.write_artifact(dongle_id, route_id, timeline, buckets)
  bus_msg = ','.join(f'{m}:{chosen_buses[m]}' for m in (HCA_03, QFK_01, LH_EPS_03))
  rs = cache.RouteStatus(
    dongle_id, route_id, 'ok', message=f'lcp_seen={has_lcp} buses={bus_msg}',
    fingerprint=carparams['fingerprint'], vin=carparams['vin'],
    can_bus=int(chosen_buses[HCA_03]),
    duration_engaged_s=float(mask_loose.sum() * features.RESAMPLE_DT),
    duration_strict_gated_s=float(mask_strict.sum() * features.RESAMPLE_DT),
  )
  cache.write_status(rs)
  return rs
