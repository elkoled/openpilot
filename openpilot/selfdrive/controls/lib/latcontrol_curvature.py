import math
import numpy as np
from collections import deque

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

LAT_ACCEL_SATURATION_THRESHOLD = 0.5  # m/s^2


class LatControlCurvature(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.sat_check_min_speed = 5.
    self.desired_curvature_buffer_len = int(1. / self.dt)
    self.desired_curvature_buffer = deque([0.] * self.desired_curvature_buffer_len, maxlen=self.desired_curvature_buffer_len)

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    curvature_log = log.ControlsState.LateralCurvatureState.new_message()
    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    error = desired_curvature - actual_curvature
    self.desired_curvature_buffer.append(desired_curvature)
    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.desired_curvature_buffer_len))
    expected_curvature = self.desired_curvature_buffer[-delay_frames]

    # the car closes the loop on the current curvature
    output_curvature = desired_curvature if active else 0.0
    curvature_log.active = active

    curvature_log.error = float(error)
    curvature_log.actualCurvature = float(actual_curvature)
    curvature_log.desiredCurvature = float(desired_curvature)
    curvature_log.output = float(output_curvature)
    curvature_log.saturated = bool(self._check_saturation(abs(expected_curvature - actual_curvature) * CS.vEgo ** 2 > LAT_ACCEL_SATURATION_THRESHOLD, CS,
                                                          False, curvature_limited))
    return 0.0, float(output_curvature), curvature_log
