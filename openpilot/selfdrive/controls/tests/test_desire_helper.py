from types import SimpleNamespace

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, MODEL_LC_TIME, LaneChangeState, LaneChangeDirection

X = np.linspace(0., 100., 33)


def line(y: float):
  return SimpleNamespace(x=X, y=np.full_like(X, y))


def model(plan_offset: float, prob: float = 0.9):
  return SimpleNamespace(laneLineProbs=[0.5, prob, prob, 0.5], laneLines=[line(-5.4), line(-1.8), line(1.8), line(5.4)],
                         position=SimpleNamespace(x=X, y=np.full_like(X, plan_offset)))


def carstate(v: float = 25., right_blindspot: bool = False):
  return car.CarState.new_message(vEgo=v, rightBlindspot=right_blindspot)


def run(dh, cs, md, seconds, lane_change_prob=0.):
  for _ in range(int(seconds / DT_MDL)):
    dh.update(cs, True, lane_change_prob, md)


class TestModelLaneChange:
  def test_drift_becomes_a_lane_change(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    assert dh.lane_change_state == LaneChangeState.laneChangeStarting
    assert dh.lane_change_direction == LaneChangeDirection.right
    assert dh.desire == log.Desire.laneChangeRight

  def test_finishes_and_waits(self):
    dh = DesireHelper()
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2, lane_change_prob=0.5)
    run(dh, carstate(), model(0.), 1., lane_change_prob=0.)
    assert dh.lane_change_state == LaneChangeState.off and dh.desire == log.Desire.none
    # another drift right away doesn't start a second one during the cooldown
    run(dh, carstate(), model(1.5), MODEL_LC_TIME + 0.2)
    assert dh.lane_change_state == LaneChangeState.off

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
