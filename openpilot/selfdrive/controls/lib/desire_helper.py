import numpy as np

from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5

# the model drifting across a lane line on its own becomes a real lane change: its plan held this far off the lane center
# this far ahead, for this long, with both lane lines clear
MODEL_LC_LOOKAHEAD = 2.5  # s
MODEL_LC_OFFSET = 1.0  # m
MODEL_LC_MAX_OFFSET = 2.5  # m, past this the lane lines and plan don't agree on the frame, ignore it
MODEL_LC_TIME = 1.0  # s
MODEL_LC_MIN_PROB = 0.6
MODEL_LC_LANE_WIDTH = (2.5, 4.8)  # m
MODEL_LC_COOLDOWN = 5.  # s after a model lane change before another
MODEL_LC_CONFIRM_TIME = 8.  # s the driver has to confirm it with the turn signal
MODEL_LC_LOST_TIME = 1.  # s the model can stop drifting before the request is dropped
MODEL_LC_DECLINED_COOLDOWN = 15.  # s after an unconfirmed request before asking again
# until the driver signals, the plan's target ahead is pulled back into the lane past this much offset
MODEL_LC_HOLD_DEADBAND = 0.5  # m
MODEL_LC_HOLD_MAX_OFFSET = 5.  # m
MODEL_LC_HOLD_GAIN = 3.  # x the curvature that moves the target back, the model re-aims from where the car is
MODEL_LC_HOLD_MAX_ACCEL = 1.  # m/s^2
MODEL_LC_HOLD_RAMP = 2.  # 1/s

class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.model_lane_change = False
    self.drift_dir = 0
    self.drift_time = 0.
    self.model_lc_cooldown = 0.
    self.drift_lost_time = 0.
    self.lane_hold_gain = 0.
    self.lane_hold_curvature = 0.

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  @staticmethod
  def plan_lane_offset(model, v_ego: float, max_offset: float = MODEL_LC_MAX_OFFSET) -> float | None:
    # the plan's offset from the ego lane center ahead, positive to the right, None when the lane isn't clear
    if model is None or min(model.laneLineProbs[1], model.laneLineProbs[2]) < MODEL_LC_MIN_PROB:
      return None
    x = max(v_ego * MODEL_LC_LOOKAHEAD, 10.)
    left = np.interp(x, model.laneLines[1].x, model.laneLines[1].y)
    right = np.interp(x, model.laneLines[2].x, model.laneLines[2].y)
    if not MODEL_LC_LANE_WIDTH[0] < right - left < MODEL_LC_LANE_WIDTH[1]:
      return None
    offset = float(np.interp(x, model.position.x, model.position.y) - (left + right) / 2)
    return offset if abs(offset) < max_offset else None

  def update_drift(self, model, v_ego: float):
    offset = self.plan_lane_offset(model, v_ego)
    direction = 0 if offset is None or abs(offset) < MODEL_LC_OFFSET else int(np.sign(offset))
    self.drift_time = self.drift_time + DT_MDL if direction and direction == self.drift_dir else 0.
    self.drift_dir = direction
    self.model_lc_cooldown = max(self.model_lc_cooldown - DT_MDL, 0.)

  def update_lane_hold(self, model, v_ego: float, holding: bool):
    # pure pursuit back to the lane: the curvature that moves the plan's target ahead by the offset past the deadband
    self.lane_hold_gain = float(np.clip(self.lane_hold_gain + (1 if holding else -1) * MODEL_LC_HOLD_RAMP * DT_MDL, 0., 1.))
    offset = self.plan_lane_offset(model, v_ego, MODEL_LC_HOLD_MAX_OFFSET)
    if offset is None or self.lane_hold_gain == 0.:
      self.lane_hold_curvature = 0.
      return
    x = max(v_ego * MODEL_LC_LOOKAHEAD, 10.)
    excess = np.sign(offset) * max(abs(offset) - MODEL_LC_HOLD_DEADBAND, 0.)
    limit = MODEL_LC_HOLD_MAX_ACCEL / max(v_ego, 1.) ** 2
    self.lane_hold_curvature = float(self.lane_hold_gain * np.clip(-2 * MODEL_LC_HOLD_GAIN * excess / x ** 2, -limit, limit))

  def update_lane_change_request(self, carstate, one_blinker: bool, below_lane_change_speed: bool):
    # the model asked for a lane change, the driver confirms it with the turn signal that way
    right = self.lane_change_direction == LaneChangeDirection.right
    blindspot = carstate.rightBlindspot if right else carstate.leftBlindspot
    if one_blinker:
      if self.get_lane_change_direction(carstate) != self.lane_change_direction:
        # signaled the other way, a normal lane change that way
        self.model_lane_change = False
        self.lane_change_direction = self.get_lane_change_direction(carstate)
      elif not blindspot and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.laneChangeStarting
        self.lane_change_timer = 0.0
      return

    self.lane_change_timer += DT_MDL
    self.drift_lost_time = 0. if self.drift_dir == (1 if right else -1) else self.drift_lost_time + DT_MDL
    steered_away = carstate.steeringPressed and (carstate.steeringTorque > 0) == right
    if (below_lane_change_speed or blindspot or steered_away or self.drift_lost_time > MODEL_LC_LOST_TIME or
        self.lane_change_timer > MODEL_LC_CONFIRM_TIME):
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
      self.lane_change_timer = 0.0
      self.model_lane_change = False
      self.model_lc_cooldown = MODEL_LC_DECLINED_COOLDOWN
      self.drift_time = 0.

  def update(self, carstate, lateral_active, lane_change_prob, model=None):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN
    self.update_drift(model, v_ego)

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
      self.lane_change_timer = 0.0
      self.model_lane_change = False
    else:
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_timer = 0.0
        # Initialize lane change direction to prevent UI alert flicker
        self.lane_change_direction = self.get_lane_change_direction(carstate)

      elif (self.lane_change_state == LaneChangeState.off and not one_blinker and not below_lane_change_speed and
            self.drift_time >= MODEL_LC_TIME and self.model_lc_cooldown == 0.):
        # the model is heading into the next lane, ask the driver to confirm a lane change, unless that side is occupied
        direction = LaneChangeDirection.right if self.drift_dir > 0 else LaneChangeDirection.left
        blindspot = carstate.rightBlindspot if self.drift_dir > 0 else carstate.leftBlindspot
        if not blindspot:
          self.lane_change_state = LaneChangeState.preLaneChange
          self.lane_change_direction = direction
          self.lane_change_timer = 0.0
          self.model_lane_change = True
          self.drift_lost_time = 0.

      elif self.lane_change_state == LaneChangeState.preLaneChange and self.model_lane_change:
        self.update_lane_change_request(carstate, one_blinker, below_lane_change_speed)

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Update lane change direction
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_timer = 0.0
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting
          self.lane_change_timer = 0.0

      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        self.lane_change_timer += DT_MDL

        if lane_change_prob < 0.02 and self.lane_change_timer >= LANE_CHANGE_START_TIME:
          self.lane_change_timer = 0.0
          if self.model_lane_change:
            # done, wait before the model can ask again
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none
            self.model_lane_change = False
            self.model_lc_cooldown = MODEL_LC_COOLDOWN
            self.drift_time = 0.
          elif one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
            self.lane_change_direction = self.get_lane_change_direction(carstate)
          else:
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none

    self.prev_one_blinker = one_blinker and lateral_active

    # the model only leaves the lane once the driver signals
    waiting = self.lane_change_state == LaneChangeState.off or (self.lane_change_state == LaneChangeState.preLaneChange and self.model_lane_change)
    self.update_lane_hold(model, v_ego, lateral_active and waiting and not one_blinker and not below_lane_change_speed)

    self.desire = log.Desire.none
    if self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
