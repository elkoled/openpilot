import unittest
from unittest.mock import patch

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.test.process_replay import model_replay


class TestModelReplay(unittest.TestCase):
  def replay(self, big, driving_times=None, monitoring_times=None, chestnut=True):
    def messages(service, times):
      result = []
      for i, execution_time in enumerate(times):
        msg = messaging.new_message(service)
        getattr(msg, service).modelExecutionTime = execution_time
        if service == "modelV2":
          msg.modelV2.big = big[i]
        result.append(msg.as_reader())
      return result

    driving = messages("modelV2", driving_times if driving_times is not None else [0.04] * len(big))
    monitoring = messages("driverStateV2", monitoring_times if monitoring_times is not None else [0.01] * 3)
    logs = [messaging.new_message(s).as_reader() for s in ("extrinsicsCalibration", "deviceState")]
    with patch.object(model_replay, "CHESTNUT", chestnut), patch.object(model_replay, "PC", False), \
         patch.object(model_replay, "trim_logs", return_value=[]), \
         patch.object(model_replay, "get_process_config", side_effect=lambda name: name), \
         patch.object(model_replay, "replay_process", side_effect=[driving, monitoring]):
      self.assertEqual(model_replay.model_replay(logs, {}), driving + monitoring)

  def test_reference_names(self):
    for chestnut, backend in ((False, "tici"), (True, "chestnut")):
      with self.subTest(chestnut=chestnut), patch.object(model_replay, "CHESTNUT", chestnut):
        self.assertEqual(model_replay.get_log_fn("route"), f"route_model_{backend}_master.zst")
        self.assertEqual(model_replay.get_log_fn("route", "commit"), f"route_model_{backend}_commit.zst")

  def test_big_model(self):
    self.replay([True] * 3)

  def test_fallback_is_rejected_in_every_frame(self):
    for big in ([False] * 3, [False, True, True], [True, False, True], [True, True, False]):
      with self.subTest(big=big), self.assertRaisesRegex(AssertionError, "without fallback"):
        self.replay(big)

  def test_missing_model_output_is_rejected(self):
    with self.assertRaisesRegex(AssertionError, "without fallback"):
      self.replay([])

  def test_chestnut_instant_timing_limit(self):
    with self.assertRaises(AssertionError):
      self.replay([True] * 3, driving_times=[0.04, 0.051, 0.04])

  def test_monitoring_average_timing_limit(self):
    with self.assertRaises(AssertionError):
      self.replay([True] * 3, monitoring_times=[0.01, 0.019, 0.019])

  def test_first_timing_sample_is_still_ignored(self):
    self.replay([True] * 3, driving_times=[1.0, 0.04, 0.04], monitoring_times=[1.0, 0.01, 0.01])

  def test_legacy_small_model(self):
    self.replay([False] * 3, driving_times=[0.02] * 3, chestnut=False)

  def test_legacy_small_model_timing_limit(self):
    with self.assertRaises(AssertionError):
      self.replay([False] * 3, driving_times=[0.04] * 3, chestnut=False)

  def test_legacy_chestnut_auto_detection(self):
    self.replay([True] * 3, chestnut=False)


if __name__ == "__main__":
  unittest.main()
