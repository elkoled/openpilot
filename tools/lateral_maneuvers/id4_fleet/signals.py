"""Per-route signal feed: cereal messages + DBC-parsed CAN.

Wraps LogReader so the per-route worker only sees lag-aligned (desired, actual,
conditioning) tuples at livePose ticks. No hand-rolled CAN decoding -- HCA_03,
QFK_01, LH_EPS_03 are pulled through CANParser('vw_meb').
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from cereal import car
from opendbc.can.parser import CANParser
from opendbc.car.can_definitions import CanData
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose
from openpilot.tools.lib.logreader import LogReader


CEREAL_WHICH = (
  "carParams", "carState", "carControl", "carOutput", "controlsState",
  "livePose", "liveCalibration", "liveDelay", "liveParameters", "can",
)


def _safe(obj, name: str, default):
  try:
    return getattr(obj, name)
  except Exception:
    return default


@dataclass
class Sample:
  """One snapshot at a livePose tick. Both 'desired' fields are already lag-adjusted upstream
  by controlsd (see ControlsState.desiredCurvature: 'lag adjusted curvatures used by lateral
  controllers'), so this tool samples desired and actual at the same t. lateral_delay is recorded
  separately as a conditioning variable so analyses can flag dongles with bad delay estimates."""
  t: float
  v_ego: float
  lat_active: bool
  steering_pressed: bool
  roll: float
  roll_compensation: float
  yaw_rate: float
  yaw_rate_std: float
  pose_valid: bool
  # desired (sampled at t; openpilot5's desiredCurvature is already lag-adjusted)
  desired_curvature: float                # carControl.actuators.curvature == controlsState.desiredCurvature
  model_curvature: float                  # controlsState.curvature (path curvature from vehicle model, pre-lag-adjustment)
  apply_curvature: float                  # carOutput.actuatorsOutput.curvature (after carcontroller additive correction)
  hca_curvature: float                    # signed, latest before t
  hca_request_status: float
  hca_power: float
  # actual
  qfk_curvature: float                    # signed, latest before t
  qfk_status: float
  # conditioning
  steer_ratio: float
  stiffness_factor: float
  lateral_delay: float
  driver_torque_nm: float
  # housekeeping
  car_fingerprint: str
  car_vin: str


@dataclass
class _Hist:
  ts: deque = field(default_factory=deque)
  vals: deque = field(default_factory=deque)
  maxlen: int = 500   # 5 s at 100 Hz

  def add(self, t: float, v) -> None:
    self.ts.append(t)
    self.vals.append(v)
    while len(self.ts) > self.maxlen:
      self.ts.popleft()
      self.vals.popleft()

  def at_or_before(self, target_t: float):
    if not self.ts or target_t < self.ts[0]:
      return None
    for i in range(len(self.ts) - 1, -1, -1):
      if self.ts[i] <= target_t:
        return self.vals[i]
    return None


def iter_samples(route_spec: str) -> Iterator[Sample]:
  """Yield one Sample per livePose tick. route_spec like 'dongle/route/a'."""
  cp = CANParser("vw_meb", [("HCA_03", 50), ("QFK_01", 50), ("LH_EPS_03", 50)], 0)

  calibrator = PoseCalibrator()
  CP: car.CarParams | None = None
  car_fingerprint = ""
  car_vin = ""
  lateral_delay = 0.0

  h_v_ego = _Hist()
  h_steering_pressed = _Hist()
  h_lat_active = _Hist()
  h_roll_comp = _Hist()
  h_desired = _Hist()
  h_model_curvature = _Hist()
  h_apply_curvature = _Hist()
  h_actuators_curvature = _Hist()
  h_steer_ratio = _Hist()
  h_stiffness = _Hist()
  h_hca_curvature = _Hist()
  h_hca_request_status = _Hist()
  h_hca_power = _Hist()
  h_qfk_curvature = _Hist()
  h_qfk_status = _Hist()
  h_driver_torque = _Hist()

  for msg in LogReader(route_spec):
    w = msg.which()
    t = msg.logMonoTime * 1e-9

    if w == "carParams":
      CP = msg.carParams
      car_fingerprint = CP.carFingerprint
      car_vin = CP.carVin
    elif w == "carState":
      h_v_ego.add(t, float(msg.carState.vEgo))
      pressed = bool(msg.carState.steeringPressed) or bool(_safe(msg.carState, "steeringSlightlyPressed", False))
      h_steering_pressed.add(t, pressed)
    elif w == "carControl":
      h_lat_active.add(t, bool(msg.carControl.latActive))
      # rollCompensation is sunnypilot-only; high-roll gate already excludes banked-road samples.
      h_roll_comp.add(t, float(_safe(msg.carControl, "rollCompensation", 0.0)))
      h_actuators_curvature.add(t, float(msg.carControl.actuators.curvature))
    elif w == "carOutput":
      h_apply_curvature.add(t, float(msg.carOutput.actuatorsOutput.curvature))
    elif w == "controlsState":
      h_desired.add(t, float(msg.controlsState.desiredCurvature))
      h_model_curvature.add(t, float(msg.controlsState.curvature))
    elif w == "liveCalibration":
      calibrator.feed_live_calib(msg.liveCalibration)
    elif w == "liveDelay":
      lateral_delay = float(msg.liveDelay.lateralDelay)
    elif w == "liveParameters":
      h_steer_ratio.add(t, float(msg.liveParameters.steerRatio))
      h_stiffness.add(t, float(msg.liveParameters.stiffnessFactor))
    elif w == "can":
      frames = [CanData(c.address, c.dat, c.src) for c in msg.can]
      if frames:
        cp.update([(msg.logMonoTime, frames)])
      hca = cp.vl["HCA_03"]
      qfk = cp.vl["QFK_01"]
      eps = cp.vl["LH_EPS_03"]
      hca_sign = -1.0 if float(hca["Curvature_VZ"]) > 0.5 else 1.0
      qfk_sign = -1.0 if float(qfk["Curvature_VZ"]) > 0.5 else 1.0
      eps_sign = -1.0 if float(eps["EPS_VZ_Lenkmoment"]) > 0.5 else 1.0
      h_hca_curvature.add(t, hca_sign * float(hca["Curvature"]))
      h_hca_request_status.add(t, float(hca["RequestStatus"]))
      h_hca_power.add(t, float(hca["Power"]))
      h_qfk_curvature.add(t, qfk_sign * float(qfk["Curvature"]))
      h_qfk_status.add(t, float(qfk["LatCon_HCA_Status"]))
      h_driver_torque.add(t, eps_sign * float(eps["EPS_Lenkmoment"]) / 100.0)   # cNm -> Nm
    elif w == "livePose":
      lp = msg.livePose
      v_ego = h_v_ego.at_or_before(t)
      lat_active = h_lat_active.at_or_before(t)
      roll_comp = h_roll_comp.at_or_before(t)
      steering_pressed = h_steering_pressed.at_or_before(t)
      desired = h_desired.at_or_before(t)
      if desired is None:
        # fall back to actuators.curvature (same source path) for stock-TA segments
        # that come up before the first controlsState
        desired = h_actuators_curvature.at_or_before(t)
      model_curv = h_model_curvature.at_or_before(t)
      apply_c = h_apply_curvature.at_or_before(t)
      steer_ratio = h_steer_ratio.at_or_before(t)
      stiffness = h_stiffness.at_or_before(t)
      hca_c = h_hca_curvature.at_or_before(t)
      hca_req = h_hca_request_status.at_or_before(t)
      hca_pwr = h_hca_power.at_or_before(t)
      qfk_c = h_qfk_curvature.at_or_before(t)
      qfk_st = h_qfk_status.at_or_before(t)
      driver_torque = h_driver_torque.at_or_before(t)
      pose_valid = bool(lp.angularVelocityDevice.valid and lp.posenetOK and lp.inputsOK and calibrator.calib_valid)

      if v_ego is None or lat_active is None or desired is None or apply_c is None or steering_pressed is None:
        continue

      device_pose = Pose.from_live_pose(msg.livePose)
      try:
        cal_pose = calibrator.build_calibrated_pose(device_pose)
        yaw_rate = float(cal_pose.angular_velocity.yaw)
        yaw_rate_std = float(cal_pose.angular_velocity.yaw_std)
      except Exception:
        yaw_rate = float(device_pose.angular_velocity.yaw)
        yaw_rate_std = float(device_pose.angular_velocity.yaw_std)
      roll = float(device_pose.orientation.roll)

      yield Sample(
        t=t,
        v_ego=float(v_ego),
        lat_active=bool(lat_active),
        steering_pressed=bool(steering_pressed),
        roll=roll,
        roll_compensation=float(roll_comp) if roll_comp is not None else 0.0,
        yaw_rate=yaw_rate,
        yaw_rate_std=yaw_rate_std,
        pose_valid=pose_valid,
        desired_curvature=float(desired),
        model_curvature=float(model_curv) if model_curv is not None else float(desired),
        apply_curvature=float(apply_c),
        hca_curvature=float(hca_c) if hca_c is not None else 0.0,
        hca_request_status=float(hca_req) if hca_req is not None else 0.0,
        hca_power=float(hca_pwr) if hca_pwr is not None else 0.0,
        qfk_curvature=float(qfk_c) if qfk_c is not None else 0.0,
        qfk_status=float(qfk_st) if qfk_st is not None else 0.0,
        steer_ratio=float(steer_ratio) if steer_ratio is not None else float("nan"),
        stiffness_factor=float(stiffness) if stiffness is not None else float("nan"),
        lateral_delay=lateral_delay,
        driver_torque_nm=float(driver_torque) if driver_torque is not None else 0.0,
        car_fingerprint=car_fingerprint,
        car_vin=car_vin,
      )
