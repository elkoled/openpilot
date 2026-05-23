"""Replication of the sunnypilot CurvatureD learner's gates, applied to streamed samples.

Mirrors ~/openpilot10/selfdrive/locationd/curvatured.py:
  - lat-accel <= 1 m/s^2
  - |sin(roll) * g| <= 0.10 m/s^2
  - yaw_rate_std < 1.0
  - vEgo >= MIN_SPEED (= SPEED_ANCHORS[0] * 0.5 = ~2.78 m/s)
  - 2 s engagement buffer after latActive rising edge
  - 2 s buffer after the last steeringPressed / steeringSlightlyPressed
  - lat_active is True at the current (lag-aligned) sample
"""
import math
from dataclasses import dataclass, field

import numpy as np


SPEED_ANCHORS_KPH = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=np.float32)
SPEED_ANCHORS = (SPEED_ANCHORS_KPH / 3.6).astype(np.float32)
MIN_SPEED = float(SPEED_ANCHORS[0] * 0.5)
MAX_LAT_ACCEL = 1.0
MAX_ROLL_LAT_ACCEL = 0.10
MAX_YAW_RATE_STD = 1.0
ENGAGE_BUFFER_S = 2.0
ACCELERATION_DUE_TO_GRAVITY = 9.81

GATE_NAMES = (
  "lat_inactive",
  "override_buffer",
  "engage_buffer",
  "low_speed",
  "high_lat_accel",
  "high_roll",
  "high_yaw_std",
  "pose_invalid",
)


@dataclass
class GateState:
  last_lat_inactive_t: float = -math.inf
  last_override_t: float = -math.inf
  rejected: dict[str, int] = field(default_factory=lambda: {n: 0 for n in GATE_NAMES})
  passed: int = 0

  def note_lat_inactive(self, t: float) -> None:
    self.last_lat_inactive_t = t

  def note_override(self, t: float) -> None:
    self.last_override_t = t

  def check(self, t: float, lat_active: bool, v_ego: float, desired_curvature: float,
            roll: float, yaw_rate_std: float, pose_valid: bool) -> str | None:
    """Return None if the sample passes all gates, otherwise the name of the first failing gate."""
    if not pose_valid:
      self.rejected["pose_invalid"] += 1
      return "pose_invalid"
    if not lat_active:
      self.rejected["lat_inactive"] += 1
      return "lat_inactive"
    if (t - self.last_override_t) < ENGAGE_BUFFER_S:
      self.rejected["override_buffer"] += 1
      return "override_buffer"
    if (t - self.last_lat_inactive_t) < ENGAGE_BUFFER_S:
      self.rejected["engage_buffer"] += 1
      return "engage_buffer"
    if v_ego < MIN_SPEED:
      self.rejected["low_speed"] += 1
      return "low_speed"
    if abs(desired_curvature) * v_ego * v_ego > MAX_LAT_ACCEL:
      self.rejected["high_lat_accel"] += 1
      return "high_lat_accel"
    if abs(math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY) > MAX_ROLL_LAT_ACCEL:
      self.rejected["high_roll"] += 1
      return "high_roll"
    if yaw_rate_std >= MAX_YAW_RATE_STD:
      self.rejected["high_yaw_std"] += 1
      return "high_yaw_std"
    self.passed += 1
    return None

  def summary(self) -> dict[str, int]:
    total = self.passed + sum(self.rejected.values())
    out = {"gate_passed": self.passed, "gate_total": total}
    out.update({f"gate_reject_{name}": count for name, count in self.rejected.items()})
    return out
