#!/usr/bin/env python3
"""
Per-route streaming extractor for lateral tracking quality on VW MEB (ID4_MK1 focus).

Streams an rlog (or qlog) once, gates samples by engagement / override / roll /
yaw-rate-uncertainty / speed, aligns desired vs actual using liveDelay.lateralDelay,
and accumulates sufficient statistics per (speed-bin x signed-curvature-bin) bucket.

Three "truth" curvature sources are tracked independently:
  - pose_yaw   : livePose.angularVelocityDevice.z / vEgo - rollCompensation
                 (sunnypilot's choice; closes loop on vehicle response)
  - cs_yaw     : carState.yawRate / vEgo                    (gyro fallback)
  - qfk_rack   : carState.curvatureMeas                     (EPS rack, already
                                                              decoded by carstate)

For each bucket the extractor emits sufficient statistics — never raw timelines —
so a few hundred routes per worker fit comfortably in RAM.

Output: NPZ file with bucket arrays plus a small JSON header.

CLI:
  python fleet_lateral_extract.py <dongle_id>/<route_id>[/a]  out.npz
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from openpilot.tools.lib.logreader import LogReader, LogsUnavailable  # noqa: E402
from opendbc.can import CANParser  # noqa: E402

DBC_NAME = "vw_meb"
QFK_01_ADDR = 0x13D   # 317  EPS rack curvature feedback
HCA_03_ADDR = 0x303   # 771  commanded curvature to rack
RELEVANT_ADDRS = frozenset((QFK_01_ADDR, HCA_03_ADDR))
PROBED_BUSES = (0, 1, 2)


def _filtered_frames(msg, which: str):
    """Yield (addr, bytes(dat), src) tuples for only the QFK_01 / HCA_03 frames.
    CANParser internally filters by its message list, but doing it here first
    avoids ~100x of Python-level work on typical CAN traffic."""
    frames = msg.can if which == "can" else msg.sendcan
    for fr in frames:
        if fr.address in RELEVANT_ADDRS:
            yield (fr.address, bytes(fr.dat), fr.src)


def _qfk_to_curvature(curv: float, vz: float) -> float:
    """Mirror MEB carstate convention (opendbc/car/volkswagen/carstate.py:257):
        curvature_meas = -QFK.Curvature * (1, -1)[int(VZ)]
    i.e.  VZ=0 -> -QFK   ;   VZ=1 -> +QFK."""
    return float(curv) if int(round(vz)) else -float(curv)


SCHEMA_VERSION = 2

# ---- Bucketing -------------------------------------------------------------

SPEED_EDGES = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 50.0], dtype=np.float64)
SIGNED_CURVATURE_EDGES = np.array([
    -1.0,
    -8.0e-3, -4.0e-3, -2.0e-3, -1.0e-3, -5.0e-4, -2.5e-4, -1.0e-4,
    +1.0e-4, +2.5e-4, +5.0e-4, +1.0e-3, +2.0e-3, +4.0e-3, +8.0e-3,
    +1.0,
], dtype=np.float64)

N_SPEED = len(SPEED_EDGES) - 1       # 7
N_CURV = len(SIGNED_CURVATURE_EDGES) - 1  # 15

TRUTH_NAMES = ["pose_yaw", "cs_yaw", "qfk_rack"]
N_TRUTH = len(TRUTH_NAMES)

# ---- Gates -----------------------------------------------------------------

MIN_VEGO = 5.0
ENGAGE_BUFFER_S = 2.0
OVERRIDE_BUFFER_S = 2.0
MAX_ROLL_LAT_ACCEL = 0.10
MAX_YAW_RATE_STD = 1.0
ACCELERATION_DUE_TO_GRAVITY = 9.81

# Sample tick downselect.  carState arrives at ~100Hz; we don't need every tick.
TICK_MIN_DT_NS = int(20e6)  # 20 ms => max ~50 Hz, plenty for steady-state stats


# ---- Bucket key helpers ----------------------------------------------------

def speed_bin(v: float) -> int:
    if v < SPEED_EDGES[0]:
        return -1
    idx = int(np.searchsorted(SPEED_EDGES, v, side="right") - 1)
    if idx >= N_SPEED:
        return N_SPEED - 1
    return idx


def curvature_bin(c: float) -> int:
    idx = int(np.searchsorted(SIGNED_CURVATURE_EDGES, c, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= N_CURV:
        return N_CURV - 1
    return idx


# ---- Stats accumulator -----------------------------------------------------

@dataclass
class BucketStats:
    n: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.int64))
    desired_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    desired_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    vmcurv_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    vmcurv_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    hca_sent_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    hca_sent_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    hca_sent_n: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.int64))
    vego_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))
    roll_abs_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV), dtype=np.float64))

    truth_n: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.int64))
    truth_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    truth_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_td_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_td_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_tv_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_tv_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_ht_sum: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))
    resid_ht_sumsq: np.ndarray = field(default_factory=lambda: np.zeros((N_SPEED, N_CURV, N_TRUTH), dtype=np.float64))

    def update(self, s, c, desired, vmcurv, hca_sent, vego, roll, truths) -> None:
        self.n[s, c] += 1
        self.desired_sum[s, c] += desired
        self.desired_sumsq[s, c] += desired * desired
        self.vmcurv_sum[s, c] += vmcurv
        self.vmcurv_sumsq[s, c] += vmcurv * vmcurv
        self.vego_sum[s, c] += vego
        self.roll_abs_sum[s, c] += abs(roll)
        if hca_sent is not None and math.isfinite(hca_sent):
            self.hca_sent_sum[s, c] += hca_sent
            self.hca_sent_sumsq[s, c] += hca_sent * hca_sent
            self.hca_sent_n[s, c] += 1

        for t, truth in enumerate(truths):
            if truth is None or not math.isfinite(truth):
                continue
            self.truth_n[s, c, t] += 1
            self.truth_sum[s, c, t] += truth
            self.truth_sumsq[s, c, t] += truth * truth
            r_td = truth - desired
            self.resid_td_sum[s, c, t] += r_td
            self.resid_td_sumsq[s, c, t] += r_td * r_td
            r_tv = truth - vmcurv
            self.resid_tv_sum[s, c, t] += r_tv
            self.resid_tv_sumsq[s, c, t] += r_tv * r_tv
            if hca_sent is not None and math.isfinite(hca_sent):
                r_ht = hca_sent - truth
                self.resid_ht_sum[s, c, t] += r_ht
                self.resid_ht_sumsq[s, c, t] += r_ht * r_ht


@dataclass
class RouteHeader:
    dongle_id: str = ""
    route_id: str = ""
    car_fingerprint: str = ""
    car_vin: str = ""
    git_branch: str = ""
    git_commit: str = ""
    git_remote: str = ""
    schema_version: int = SCHEMA_VERSION
    source: str = "rlog"

    duration_s: float = 0.0
    engaged_s: float = 0.0
    accepted_s: float = 0.0

    steer_ratio_mean: float = float("nan")
    steer_ratio_std: float = float("nan")
    steer_ratio_n: int = 0
    stiffness_factor_mean: float = float("nan")
    stiffness_factor_std: float = float("nan")
    stiffness_factor_n: int = 0
    lateral_delay_mean: float = float("nan")
    lateral_delay_std: float = float("nan")
    lateral_delay_n: int = 0

    has_curvature_params: bool = False
    curvature_use_params_frac: float = 0.0

    n_messages: int = 0
    n_carcontrol: int = 0
    n_lat_active: int = 0
    n_caroutput: int = 0
    n_livepose: int = 0
    qfk_bus: int = -1
    n_qfk_frames: int = 0
    notes: list[str] = field(default_factory=list)


def _running_stats_finalize(sum_x, sum_xx, n):
    if n <= 0:
        return float("nan"), float("nan")
    mean = sum_x / n
    var = max(sum_xx / n - mean * mean, 0.0)
    return mean, math.sqrt(var)


def _yaw_rate_to_curvature(yaw_rate, vego, roll_compensation):
    if not (math.isfinite(yaw_rate) and math.isfinite(vego)) or vego < 1e-3:
        return None
    return float(yaw_rate / max(vego, 0.1) - (roll_compensation if math.isfinite(roll_compensation) else 0.0))


def extract_route(route_path: str, out_path: str | Path | None = None,
                  dongle_id: str | None = None, route_id: str | None = None) -> dict:
    """Process one route or segment.

    route_path may be either:
      - a canonical openpilot route spec:  '<dongle>/<route>[/a|q|r]'
      - a direct rlog URL.  In that case pass dongle_id / route_id explicitly
        so the per-segment header is correctly populated.
    """
    header = RouteHeader()
    if dongle_id is not None:
        header.dongle_id = dongle_id
    if route_id is not None:
        header.route_id = route_id
    if not header.dongle_id or not header.route_id:
        parts = route_path.replace("|", "/").split("/")
        if len(parts) >= 2 and "://" not in route_path:
            header.dongle_id, header.route_id = parts[0], parts[1]

    t_start = time.time()
    try:
        lr = LogReader(route_path)
    except LogsUnavailable as e:
        header.notes.append(f"LogsUnavailable: {str(e)[:160]}")
        return _finalize(header, BucketStats(), out_path, t_start, ok=False)
    except Exception as e:  # noqa: BLE001
        header.notes.append(f"LogReader init failed: {type(e).__name__}: {str(e)[:160]}")
        return _finalize(header, BucketStats(), out_path, t_start, ok=False)

    bs = BucketStats()

    # CAN parsers for QFK_01 (EPS rack curvature feedback).  One per probed bus;
    # the bus that actually carries QFK_01 wins at emit time.  HCA_sent comes
    # from carOutput.actuatorsOutput.curvature so we don't need it from CAN.
    parsers: dict[int, CANParser] = {}
    for bus in PROBED_BUSES:
        try:
            parsers[bus] = CANParser(DBC_NAME, [("QFK_01", 0)], bus)
        except Exception:  # noqa: BLE001
            pass
    bus_qfk_count = {b: 0 for b in parsers}

    # History buffer for lat-delay alignment.
    HIST_S = 1.0
    hist: deque = deque()

    # Latest state.
    last_lat_active = False
    last_roll_comp = 0.0
    last_lat_inactive_t_ns = -1
    last_lat_active_t_ns = -1
    last_override_t_ns = -1

    last_vego = float("nan")
    last_steering_pressed = False
    last_steering_slight = False
    last_cs_yaw = float("nan")

    last_pose_yaw = float("nan")
    last_pose_roll = float("nan")
    last_pose_yaw_std = float("nan")
    last_pose_valid = False
    last_pose_ok = False
    # Calibrated values surfaced on carControl (set by controlsd from
    # PoseCalibrator + livePose).  Preferred over raw livePose because they're
    # already in the road frame.
    last_cc_roll = float("nan")
    last_cc_yaw_rate = float("nan")

    last_hca_sent = float("nan")
    last_qfk_curvature = float("nan")

    last_lat_delay = 0.18

    sr_sum = sr_sumsq = 0.0; sr_n = 0
    sf_sum = sf_sumsq = 0.0; sf_n = 0
    ld_sum = ld_sumsq = 0.0; ld_n = 0

    has_curv_params = False
    curv_total = curv_active = 0

    n_msgs = n_cc = n_cc_active = n_co = n_lp = 0
    duration_ns = 0
    engaged_ns = 0
    prev_msg_t_ns = 0
    last_carcontrol_t_ns = 0
    last_emit_t_ns = 0

    car_fp = car_vin = ""
    git_branch = git_commit = git_remote = ""

    try:
        for msg in lr:
            n_msgs += 1
            which = msg.which()
            t_ns = msg.logMonoTime
            if prev_msg_t_ns:
                delta = t_ns - prev_msg_t_ns
                # Cap to 0.5s to ignore log-segment / boot gaps and out-of-order resets.
                if 0 < delta < int(5e8):
                    duration_ns += delta
            prev_msg_t_ns = t_ns

            if which == "carParams":
                try:
                    car_fp = msg.carParams.carFingerprint or car_fp
                    car_vin = msg.carParams.carVin or car_vin
                except Exception:  # noqa: BLE001
                    pass

            elif which == "initData":
                try:
                    git_branch = msg.initData.gitBranch or git_branch
                    git_commit = msg.initData.gitCommit or git_commit
                    git_remote = msg.initData.gitRemote or git_remote
                except Exception:  # noqa: BLE001
                    pass

            elif which == "liveParameters":
                try:
                    sr = float(msg.liveParameters.steerRatio)
                    if math.isfinite(sr):
                        sr_sum += sr; sr_sumsq += sr * sr; sr_n += 1
                except Exception:  # noqa: BLE001
                    pass
                try:
                    sf = float(msg.liveParameters.stiffnessFactor)
                    if math.isfinite(sf):
                        sf_sum += sf; sf_sumsq += sf * sf; sf_n += 1
                except Exception:  # noqa: BLE001
                    pass

            elif which == "liveDelay":
                try:
                    ld = float(msg.liveDelay.lateralDelay)
                    if math.isfinite(ld):
                        last_lat_delay = ld
                        ld_sum += ld; ld_sumsq += ld * ld; ld_n += 1
                except Exception:  # noqa: BLE001
                    pass

            elif which == "liveCurvatureParameters":
                has_curv_params = True
                try:
                    curv_total += 1
                    if bool(msg.liveCurvatureParameters.useParams):
                        curv_active += 1
                except Exception:  # noqa: BLE001
                    pass

            elif which == "carControl":
                n_cc += 1
                try:
                    cc = msg.carControl
                    desired_curv = float(cc.actuators.curvature)
                    current_curv = float(cc.currentCurvature)
                    new_lat_active = bool(cc.latActive)
                except Exception:  # noqa: BLE001
                    continue
                # rollCompensation only exists on sunnypilot's carControl; default 0
                # otherwise (the roll gate filters out cambered roads anyway).
                try:
                    roll_comp = float(cc.rollCompensation)
                except Exception:  # noqa: BLE001
                    roll_comp = 0.0
                # CC.orientationNED and CC.angularVelocity are calibrated by
                # controlsd (controlsd.py:151) — they're in the road frame, so
                # the roll gate works directly without us doing our own
                # calibration math.
                try:
                    o = cc.orientationNED
                    last_cc_roll = float(o[0]) if len(o) >= 1 else float("nan")
                except Exception:  # noqa: BLE001
                    last_cc_roll = float("nan")
                try:
                    a = cc.angularVelocity
                    last_cc_yaw_rate = float(a[2]) if len(a) >= 3 else float("nan")
                except Exception:  # noqa: BLE001
                    last_cc_yaw_rate = float("nan")

                # Engagement seconds (latActive duration).  Integrate as time
                # between consecutive carControl ticks while latActive was true.
                if last_carcontrol_t_ns and last_lat_active and new_lat_active:
                    delta = t_ns - last_carcontrol_t_ns
                    if 0 < delta < int(5e8):  # cap to 0.5s to ignore log gaps
                        engaged_ns += delta

                if new_lat_active:
                    n_cc_active += 1
                    last_lat_active_t_ns = t_ns
                else:
                    last_lat_inactive_t_ns = t_ns

                last_lat_active = new_lat_active
                last_roll_comp = roll_comp
                last_carcontrol_t_ns = t_ns

                hist.append((t_ns, desired_curv, current_curv, roll_comp, new_lat_active))
                cutoff = t_ns - int(HIST_S * 1e9)
                while hist and hist[0][0] < cutoff:
                    hist.popleft()

            elif which == "carOutput":
                n_co += 1
                try:
                    last_hca_sent = float(msg.carOutput.actuatorsOutput.curvature)
                except Exception:  # noqa: BLE001
                    pass

            elif which == "carState":
                try:
                    cs = msg.carState
                    last_vego = float(cs.vEgo)
                    last_steering_pressed = bool(cs.steeringPressed)
                    last_steering_slight = bool(getattr(cs, "steeringSlightlyPressed", False))
                    last_cs_yaw = float(cs.yawRate)
                except Exception:  # noqa: BLE001
                    continue
                if last_steering_pressed or last_steering_slight:
                    last_override_t_ns = t_ns

                # Tick: emit a sample if enough time has passed since last emit.
                if (t_ns - last_emit_t_ns) >= TICK_MIN_DT_NS:
                    if _try_emit_sample(
                        bs, hist, t_ns,
                        last_lat_active_t_ns, last_lat_inactive_t_ns, last_override_t_ns,
                        last_steering_pressed, last_steering_slight, last_lat_active,
                        last_vego, last_cc_yaw_rate, last_cc_roll,
                        last_pose_yaw, last_pose_yaw_std, last_pose_roll,
                        last_pose_valid, last_pose_ok, last_cs_yaw, last_qfk_curvature,
                        last_hca_sent, last_lat_delay,
                    ):
                        last_emit_t_ns = t_ns

            elif which == "can":
                # Route QFK_01 frames to the matching-bus parser only.
                frames_by_bus: dict[int, list] = {}
                for fr in msg.can:
                    if fr.address != QFK_01_ADDR:
                        continue
                    src = fr.src
                    if src not in parsers:
                        continue
                    frames_by_bus.setdefault(src, []).append((fr.address, bytes(fr.dat), src))
                    bus_qfk_count[src] += 1
                for src, frame_list in frames_by_bus.items():
                    try:
                        parsers[src].update([(t_ns, frame_list)], False)
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        q = parsers[src].vl["QFK_01"]["Curvature"]
                        qvz = parsers[src].vl["QFK_01"]["Curvature_VZ"]
                        last_qfk_curvature = _qfk_to_curvature(q, qvz)
                    except Exception:  # noqa: BLE001
                        pass

            elif which == "livePose":
                n_lp += 1
                try:
                    lp = msg.livePose
                    last_pose_yaw = float(lp.angularVelocityDevice.z)
                    last_pose_yaw_std = float(lp.angularVelocityDevice.zStd)
                    last_pose_roll = float(lp.orientationNED.x)
                    last_pose_valid = bool(lp.angularVelocityDevice.valid)
                    # Be more permissive than realtime locationd: qlog-fallback
                    # segments often have inputsOK=False; the .valid flag on the
                    # angular velocity itself is the load-bearing check.
                    last_pose_ok = bool(lp.angularVelocityDevice.valid) and bool(lp.posenetOK)
                except Exception:  # noqa: BLE001
                    pass

    except Exception as e:  # noqa: BLE001
        header.notes.append(f"stream failed at msg {n_msgs}: {type(e).__name__}: {str(e)[:160]}")

    header.n_messages = n_msgs
    header.n_carcontrol = n_cc
    header.n_lat_active = n_cc_active
    header.n_caroutput = n_co
    header.n_livepose = n_lp
    header.duration_s = duration_ns / 1e9
    header.engaged_s = engaged_ns / 1e9
    header.accepted_s = float(bs.n.sum()) * (TICK_MIN_DT_NS / 1e9)
    header.car_fingerprint = car_fp
    header.car_vin = car_vin
    header.git_branch = git_branch
    header.git_commit = git_commit
    header.git_remote = git_remote

    header.steer_ratio_mean, header.steer_ratio_std = _running_stats_finalize(sr_sum, sr_sumsq, sr_n)
    header.steer_ratio_n = sr_n
    header.stiffness_factor_mean, header.stiffness_factor_std = _running_stats_finalize(sf_sum, sf_sumsq, sf_n)
    header.stiffness_factor_n = sf_n
    header.lateral_delay_mean, header.lateral_delay_std = _running_stats_finalize(ld_sum, ld_sumsq, ld_n)
    header.lateral_delay_n = ld_n
    header.has_curvature_params = has_curv_params
    header.curvature_use_params_frac = (curv_active / curv_total) if curv_total else 0.0

    if bus_qfk_count:
        winner = max(bus_qfk_count.items(), key=lambda kv: kv[1])
        header.qfk_bus = winner[0] if winner[1] > 0 else -1
        header.n_qfk_frames = winner[1]

    return _finalize(header, bs, out_path, t_start, ok=True)


def _try_emit_sample(
    bs: BucketStats, hist, t_ns,
    last_lat_active_t_ns, last_lat_inactive_t_ns, last_override_t_ns,
    steering_pressed, steering_slight, lat_active_now,
    vego, cc_yaw_rate, cc_roll,
    pose_yaw, pose_yaw_std, pose_roll, pose_valid, pose_ok,
    cs_yaw, qfk_curvature, hca_sent, lat_delay,
) -> bool:
    if not lat_active_now:
        return False
    if steering_pressed or steering_slight:
        return False
    if last_lat_inactive_t_ns >= 0 and (t_ns - last_lat_inactive_t_ns) < ENGAGE_BUFFER_S * 1e9:
        return False
    if last_override_t_ns >= 0 and (t_ns - last_override_t_ns) < OVERRIDE_BUFFER_S * 1e9:
        return False
    if not math.isfinite(vego) or vego < MIN_VEGO:
        return False

    s = speed_bin(vego)
    if s < 0:
        return False

    target_t_ns = t_ns - int(max(lat_delay, 0.0) * 1e9)
    desired = vmcurv = None
    roll_comp = 0.0
    for entry in reversed(hist):
        if entry[0] <= target_t_ns:
            _, desired, vmcurv, roll_comp, _la = entry
            break
    if desired is None or vmcurv is None:
        return False
    if not (math.isfinite(desired) and math.isfinite(vmcurv)):
        return False

    c = curvature_bin(desired)

    # Roll gate.  Prefer the calibrated CC roll (already in road frame); fall
    # back to raw livePose roll if CC roll is missing.  Skip the gate entirely
    # if neither is available — the lat-accel envelope is still gated by the
    # speed/curvature bucketing.
    eff_roll = cc_roll if math.isfinite(cc_roll) else pose_roll
    if math.isfinite(eff_roll):
        if abs(math.sin(eff_roll)) * ACCELERATION_DUE_TO_GRAVITY > MAX_ROLL_LAT_ACCEL:
            return False

    # truth_pose: prefer calibrated CC yaw rate (always road frame).  Fall back
    # to raw livePose yaw rate (which has the calibrator-applied tilt issue
    # but is gated by pose_ok).
    truth_pose = None
    if math.isfinite(cc_yaw_rate) and cc_yaw_rate != 0.0:
        truth_pose = _yaw_rate_to_curvature(cc_yaw_rate, vego, roll_comp)
    elif pose_valid and pose_ok and math.isfinite(pose_yaw) and pose_yaw_std < MAX_YAW_RATE_STD:
        truth_pose = _yaw_rate_to_curvature(pose_yaw, vego, roll_comp)

    # carState.yawRate is not populated on VW MEB (always 0.0); treat exact zero
    # as missing so the cs_yaw truth source becomes empty on VW rather than
    # contaminating residual stats with a constant-zero "truth".
    truth_cs = _yaw_rate_to_curvature(cs_yaw, vego, 0.0) if (math.isfinite(cs_yaw) and cs_yaw != 0.0) else None

    truth_qfk = float(qfk_curvature) if math.isfinite(qfk_curvature) else None

    bs.update(
        s, c,
        desired=float(desired),
        vmcurv=float(vmcurv),
        hca_sent=hca_sent if (hca_sent is not None and math.isfinite(hca_sent)) else None,
        vego=float(vego),
        roll=float(pose_roll) if math.isfinite(pose_roll) else 0.0,
        truths=(truth_pose, truth_cs, truth_qfk),
    )
    return True


def _header_to_dict(h: RouteHeader) -> dict:
    return {k: getattr(h, k) for k in (
        "dongle_id", "route_id", "car_fingerprint", "car_vin",
        "git_branch", "git_commit", "git_remote",
        "schema_version", "source",
        "duration_s", "engaged_s", "accepted_s",
        "steer_ratio_mean", "steer_ratio_std", "steer_ratio_n",
        "stiffness_factor_mean", "stiffness_factor_std", "stiffness_factor_n",
        "lateral_delay_mean", "lateral_delay_std", "lateral_delay_n",
        "has_curvature_params", "curvature_use_params_frac",
        "n_messages", "n_carcontrol", "n_lat_active", "n_caroutput", "n_livepose",
        "qfk_bus", "n_qfk_frames",
        "notes",
    )}


def _finalize(h: RouteHeader, bs: BucketStats, out_path, t_start: float, ok: bool) -> dict:
    elapsed = time.time() - t_start
    h.notes.append(f"elapsed={elapsed:.2f}s ok={ok} accepted={int(bs.n.sum())}")
    payload = {
        "header_json": json.dumps(_header_to_dict(h)),
        "speed_edges": SPEED_EDGES,
        "curvature_edges": SIGNED_CURVATURE_EDGES,
        "truth_names": np.array(TRUTH_NAMES, dtype=object),
        "n": bs.n,
        "desired_sum": bs.desired_sum,
        "desired_sumsq": bs.desired_sumsq,
        "vmcurv_sum": bs.vmcurv_sum,
        "vmcurv_sumsq": bs.vmcurv_sumsq,
        "hca_sent_sum": bs.hca_sent_sum,
        "hca_sent_sumsq": bs.hca_sent_sumsq,
        "hca_sent_n": bs.hca_sent_n,
        "vego_sum": bs.vego_sum,
        "roll_abs_sum": bs.roll_abs_sum,
        "truth_n": bs.truth_n,
        "truth_sum": bs.truth_sum,
        "truth_sumsq": bs.truth_sumsq,
        "resid_td_sum": bs.resid_td_sum,
        "resid_td_sumsq": bs.resid_td_sumsq,
        "resid_tv_sum": bs.resid_tv_sum,
        "resid_tv_sumsq": bs.resid_tv_sumsq,
        "resid_ht_sum": bs.resid_ht_sum,
        "resid_ht_sumsq": bs.resid_ht_sumsq,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **payload)
    return payload


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: fleet_lateral_extract.py <dongle_id>/<route_id>[/a] [out.npz]", file=sys.stderr)
        return 1
    route = argv[1]
    if not route.endswith(("/a", "/r", "/q")):
        route = route.rstrip("/") + "/a"
    out = argv[2] if len(argv) >= 3 else None
    res = extract_route(route, out)
    hdr = json.loads(res["header_json"])
    accepted = int(res["n"].sum())
    print(f"{route}  fp={hdr['car_fingerprint']}  branch={hdr['git_branch']}  "
          f"dur={hdr['duration_s']:.1f}s  engaged={hdr['engaged_s']:.1f}s  "
          f"n_cc={hdr['n_carcontrol']}  n_lat_active={hdr['n_lat_active']}  "
          f"n_co={hdr['n_caroutput']}  n_lp={hdr['n_livepose']}  "
          f"qfk_bus={hdr['qfk_bus']} qfk_frames={hdr['n_qfk_frames']}  "
          f"accepted={accepted}")
    print("notes:", hdr["notes"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
