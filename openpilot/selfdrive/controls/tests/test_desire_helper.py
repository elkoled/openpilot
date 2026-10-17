from types import SimpleNamespace

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, MODEL_LC_TIME, MODEL_LC_CONFIRM_TIME, MODEL_LC_DECLINED_COOLDOWN, \
                                                         LaneChangeState, LaneChangeDirection

X = np.linspace(0., 100., 33)


def line(y: float):
  return SimpleNamespace(x=X, y=np.full_like(X, y))


def model(plan_offset: float, prob: float = 0.9):
  return SimpleNamespace(laneLineProbs=[0.5, prob, prob, 0.5], laneLines=[line(-5.4), line(-1.8), line(1.8), line(5.4)],
                         position=SimpleNamespace(x=X, y=np.full_like(X, plan_offset)))


def carstate(v: float = 25., right_blindspot: bool = False, left_blinker: bool = False, right_blinker: bool = False, torque: float = 0.):
  return car.CarState.new_message(vEgo=v, rightBlindspot=right_blindspot, leftBlinker=left_blinker, rightBlinker=right_blinker,
                                  steeringPressed=torque != 0., steeringTorque=torque)


def run(dh, cs, md, seconds, lane_change_prob=0.):
  for _ in range(int(seconds / DT_MDL)):
    dh.update(cs, True, lane_change_prob, md)


class TestModelLaneChange:
  def test_drift_asks_for_the_turn_signal(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.lane_change_direction == LaneChangeDirection.right
    assert dh.desire == log.Desire.none
    # no lane change without the driver signaling
    run(dh, carstate(), model(1.5), 2.)
    assert dh.lane_change_state == LaneChangeState.preLaneChange and dh.desire == log.Desire.none

  def test_turn_signal_confirms(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(right_blinker=True), model(1.5), DT_MDL)
    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert dh.desire == log.Desire.laneChangeRight

  def test_finishes_and_waits(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(right_blinker=True), model(1.5), 1., lane_change_prob=0.5)
    run(dh, carstate(right_blinker=True), model(0.), 1., lane_change_prob=0.)
    assert dh.lane_change_state == LaneChangeState.off and dh.desire == log.Desire.none
    # the signal still on doesn't start a second one, nor another drift during the cooldown
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    assert dh.lane_change_state == LaneChangeState.off

  def test_not_confirmed(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(), model(1.5), MODEL_LC_CONFIRM_TIME + 0.2)
    assert dh.lane_change_state == LaneChangeState.off and dh.desire == log.Desire.none
    # doesn't ask again right away
    run(dh, carstate(), model(1.5), MODEL_LC_DECLINED_COOLDOWN - 1.)
    assert dh.lane_change_state == LaneChangeState.off
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 1.2)
    assert dh.lane_change_state == LaneChangeState.preLaneChange

  def test_drift_gone_or_steered_away_drops_it(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(), model(0.), 0.5)
    assert dh.lane_change_state == LaneChangeState.preLaneChange
    run(dh, carstate(), model(0.), 1.)
    assert dh.lane_change_state == LaneChangeState.off

    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(torque=1.), model(1.5), DT_MDL)
    assert dh.lane_change_state == LaneChangeState.off

  def test_other_turn_signal_is_a_normal_lane_change(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(left_blinker=True), model(1.5), 1.)
    assert dh.lane_change_state == LaneChangeState.preLaneChange
    assert dh.lane_change_direction == LaneChangeDirection.left and not dh.model_lane_change
    run(dh, carstate(left_blinker=True, torque=1.), model(1.5), DT_MDL)
    assert dh.desire == log.Desire.laneChangeLeft

  def test_blindspot_waits_for_a_gap(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    run(dh, carstate(right_blinker=True, right_blindspot=True), model(1.5), 1.)
    assert dh.lane_change_state == LaneChangeState.preLaneChange
    run(dh, carstate(right_blinker=True), model(1.5), DT_MDL)
    assert dh.desire == log.Desire.laneChangeRight

  def test_blindspot_keeps_the_lane(self):
    dh = DesireHelper()
    run(dh, carstate(right_blindspot=True), model(1.5), MODEL_LC_TIME + 0.5)
    assert dh.lane_change_state == LaneChangeState.off

  def test_short_or_small_drift_ignored(self):
    dh = DesireHelper()
    run(dh, carstate(), model(0.6), 3.)
    assert dh.lane_change_state == LaneChangeState.off
    run(dh, carstate(), model(1.5), MODEL_LC_TIME / 2)
    run(dh, carstate(), model(0.), 0.2)
    run(dh, carstate(), model(1.5), MODEL_LC_TIME / 2)
    assert dh.lane_change_state == LaneChangeState.off

  def test_unclear_lane_or_slow_ignored(self):
    dh = DesireHelper()
    run(dh, carstate(), model(106.), 3.)
    assert dh.lane_change_state == LaneChangeState.off
    run(dh, carstate(), model(1.5, prob=0.3), 3.)
    assert dh.lane_change_state == LaneChangeState.off
    run(dh, carstate(v=5.), model(1.5), 3.)
    assert dh.lane_change_state == LaneChangeState.off
