from machine import Pin
from time import sleep_ms, ticks_add, ticks_diff, ticks_ms


class XL2515:
  RESET = 0xC0
  READ = 0x03
  WRITE = 0x02
  BIT_MODIFY = 0x05

  CANSTAT = 0x0E
  CANCTRL = 0x0F
  CNF3 = 0x28
  CNF2 = 0x29
  CNF1 = 0x2A
  CANINTE = 0x2B
  CANINTF = 0x2C
  EFLG = 0x2D
  TXB0CTRL = 0x30
  TXB0SIDH = 0x31
  TXB0SIDL = 0x32
  TXB0DLC = 0x35
  TXB0D0 = 0x36
  RXB0CTRL = 0x60
  RXB0SIDH = 0x61
  RXB0SIDL = 0x62
  RXB0DLC = 0x65
  RXB0D0 = 0x66

  def __init__(self, spi, cs_pin=9, int_pin=8):
    self.spi = spi
    self.cs = Pin(cs_pin, Pin.OUT, value=1)
    self.interrupt = Pin(int_pin, Pin.IN, Pin.PULL_UP)

  def _transaction(self, command, read_count=0):
    self.cs(0)
    try:
      self.spi.write(command)
      return self.spi.read(read_count, 0) if read_count else None
    finally:
      self.cs(1)

  def write(self, register, values):
    if isinstance(values, int):
      values = bytes((values,))
    self._transaction(bytes((self.WRITE, register)) + values)

  def read(self, register, count=1):
    return self._transaction(bytes((self.READ, register)), count)

  def bit_modify(self, register, mask, value):
    self._transaction(bytes((self.BIT_MODIFY, register, mask, value)))

  def init_500k(self):
    self._transaction(bytes((self.RESET,)))
    sleep_ms(10)
    # Waveshare XL2515 500-kbit/s timing table.
    self.write(self.CNF1, 0x00)
    self.write(self.CNF2, 0x9E)
    self.write(self.CNF3, 0x03)
    # Exact standard-ID filter for command 0x200 in RXB0.
    self.write(0x00, 0x40)
    self.write(0x01, 0x00)
    self.write(0x20, 0xFF)
    self.write(0x21, 0xE0)
    self.write(self.RXB0CTRL, 0x00)
    self.write(self.CANINTF, 0)
    self.write(self.CANINTE, 1)
    self.write(self.CANCTRL, 0)
    deadline = ticks_add(ticks_ms(), 20)
    while self.read(self.CANSTAT)[0] & 0xE0:
      if ticks_diff(deadline, ticks_ms()) <= 0:
        return False
    return True

  def receive(self):
    flags = self.read(self.CANINTF)[0]
    if not flags & 1:
      return None
    sid = self.read(self.RXB0SIDH, 2)
    can_id = (sid[0] << 3) | (sid[1] >> 5)
    length = min(self.read(self.RXB0DLC)[0] & 0x0F, 8)
    data = self.read(self.RXB0D0, length)
    self.bit_modify(self.CANINTF, 1, 0)
    return can_id, data

  def send(self, can_id, data):
    deadline = ticks_add(ticks_ms(), 3)
    while self.read(self.TXB0CTRL)[0] & 0x08:
      if ticks_diff(deadline, ticks_ms()) <= 0:
        return False
    self.write(self.TXB0SIDH, can_id >> 3)
    self.write(self.TXB0SIDL, (can_id & 7) << 5)
    self.write(self.TXB0DLC, len(data))
    self.write(self.TXB0D0, data)
    self.bit_modify(self.TXB0CTRL, 0x08, 0x08)
    return True

  def error_flags(self):
    return self.read(self.EFLG)[0]
