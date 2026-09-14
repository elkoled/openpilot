import pathlib
import sys
import types
import unittest


class Pin:
  OUT = 1
  IN = 0
  PULL_UP = 1

  def __init__(self, number, *args, **kwargs):
    self.number = number

  def __call__(self, value):
    pass


real_time = sys.modules["time"]
machine = types.ModuleType("machine")
machine.Pin = Pin
sys.modules["machine"] = machine
fake_time = types.ModuleType("time")
fake_time.sleep_ms = lambda _: None
fake_time.ticks_ms = lambda: 0
fake_time.ticks_add = lambda value, delta: value + delta
fake_time.ticks_diff = lambda a, b: a - b
sys.modules["time"] = fake_time
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from xl2515 import XL2515
sys.modules["time"] = real_time
del sys.modules["machine"]


class FakeSPI:
  def __init__(self, reads=()):
    self.writes = []
    self.reads = list(reads)

  def write(self, data):
    self.writes.append(bytes(data))

  def read(self, count, fill):
    value = self.reads.pop(0)
    assert len(value) == count
    return value


class XL2515Test(unittest.TestCase):
  def test_receive_decodes_standard_id_and_payload(self):
    spi = FakeSPI((b"\x01", b"\x40\x00", b"\x01", b"\x05"))
    can = XL2515(spi)
    self.assertEqual((0x200, b"\x05"), can.receive())
    self.assertEqual(b"\x05\x2c\x01\x00", spi.writes[-1])

  def test_receive_returns_none_without_pending_frame(self):
    spi = FakeSPI((b"\x00",))
    self.assertIsNone(XL2515(spi).receive())

  def test_send_encodes_standard_id_and_requests_tx(self):
    spi = FakeSPI((b"\x00",))
    self.assertTrue(XL2515(spi).send(0x201, b"\x01\x02"))
    self.assertIn(b"\x02\x31\x40", spi.writes)
    self.assertIn(b"\x02\x32\x20", spi.writes)
    self.assertIn(b"\x02\x36\x01\x02", spi.writes)
    self.assertEqual(b"\x05\x30\x08\x08", spi.writes[-1])


if __name__ == "__main__":
  unittest.main()
