import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from protocol import *


class ProtocolTest(unittest.TestCase):
  def test_direct_states_apply_exactly(self):
    controller = Controller()
    for state in range(8):
      self.assertTrue(controller.apply(bytes((state,)), 100 + state))
      self.assertEqual(state, controller.state)
    self.assertEqual(8, controller.rx_count)

  def test_invalid_direct_state_does_not_change_output(self):
    controller = Controller()
    controller.apply(b"\x07", 10)
    self.assertFalse(controller.apply(b"\x08", 20))
    self.assertEqual(7, controller.state)
    self.assertEqual(10, controller.last_command_ms)

  def test_protected_known_vector_and_sequence(self):
    controller = Controller()
    frame = bytes.fromhex("50 42 01 2a 01 fe 00 a8")
    self.assertTrue(controller.apply(frame, 10))
    self.assertFalse(controller.apply(frame, 11))
    self.assertTrue(controller.faults & FAULT_STALE_SEQUENCE)

  def test_watchdog_releases_after_200_ms(self):
    controller = Controller()
    controller.apply(b"\x01", 0xFFFFFF9C)
    self.assertFalse(controller.poll_watchdog(99))
    self.assertTrue(controller.poll_watchdog(100))
    self.assertEqual(0, controller.state)

  def test_status_contract(self):
    controller = Controller()
    controller.apply(b"\x02", 1)
    controller.rx_count = 0x1234
    status = controller.status()
    self.assertEqual(bytes.fromhex("50 42 01 00 02 34 12"), status[:7])
    self.assertEqual(crc8(status[:7]), status[7])


if __name__ == "__main__":
  unittest.main()
