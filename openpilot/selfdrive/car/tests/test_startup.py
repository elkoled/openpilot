import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from opendbc.car.structs import CarControl
from openpilot.cereal import log
from openpilot.selfdrive.car.card import Car


class TestStartup(unittest.TestCase):
  def setUp(self):
    self.car = Car.__new__(Car)
    self.car.CP = SimpleNamespace(passive=False)
    self.car.CI = Mock()
    self.car.CI.apply.return_value = (CarControl.Actuators(), [])
    self.car.params = Mock()
    self.car.pm = Mock()
    self.car.can_callbacks = (Mock(), Mock())
    self.car.can_log_mono_time = 0
    self.car.initialized_prev = False
    self.car.state_update = Mock(return_value=(SimpleNamespace(canValid=True), None))
    self.car.state_publish = Mock()
    self.car.sm = MagicMock()
    self.car.sm.seen = {'onroadEvents': False, 'carControl': False}
    self.live = False
    self.events = []
    self.car.sm.all_alive.side_effect = lambda _: self.live
    self.car.sm.__getitem__.side_effect = lambda key: self.events if key == 'onroadEvents' else CarControl()

  def events_ready(self):
    self.car.sm.seen['onroadEvents'] = True
    self.events = []

  def controls_ready(self):
    self.car.sm.seen['carControl'] = True
    self.live = True

  def assert_not_taken_over(self):
    self.car.CI.init.assert_not_called()
    self.car.CI.apply.assert_not_called()
    self.car.params.put_bool.assert_not_called()
    self.assertFalse(self.car.initialized_prev)

  def test_timeout_without_controlsd(self):
    self.events_ready()
    for _ in range(10):
      self.car.step()
    self.assert_not_taken_over()
    self.assertEqual(self.car.state_publish.call_count, 10)
    self.controls_ready()
    self.car.step()
    self.car.CI.init.assert_called_once()
    self.car.params.put_bool.assert_called_once_with('ControlsReady', True)
    self.car.CI.apply.assert_called_once()

  def test_waits_for_initialization_event(self):
    self.controls_ready()
    self.car.step()
    self.assert_not_taken_over()
    self.car.sm.seen['onroadEvents'] = True
    self.events = [SimpleNamespace(name=log.OnroadEvent.EventName.selfdriveInitializing)]
    self.car.step()
    self.assert_not_taken_over()

  def test_stale_first_control_message(self):
    self.events_ready()
    self.car.sm.seen['carControl'] = True
    self.car.step()
    self.assert_not_taken_over()
    self.controls_ready()
    self.car.step()
    self.car.CI.init.assert_called_once()

  def test_controlsd_stall_does_not_repeat_initialization(self):
    self.events_ready()
    self.controls_ready()
    self.car.step()
    self.live = False
    self.car.step()
    self.assertTrue(self.car.initialized_prev)
    self.car.CI.apply.assert_called_once()
    self.live = True
    self.car.step()
    self.car.CI.init.assert_called_once()
    self.car.params.put_bool.assert_called_once_with('ControlsReady', True)
    self.assertEqual(self.car.CI.apply.call_count, 2)

  def test_passive_does_not_take_over(self):
    self.events_ready()
    self.controls_ready()
    self.car.CP.passive = True
    self.car.step()
    self.car.CI.init.assert_not_called()
    self.car.params.put_bool.assert_not_called()


if __name__ == '__main__':
  unittest.main()
