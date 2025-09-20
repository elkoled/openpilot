# must be built with scons
from msgq.ipc_pyx import Context, Poller, SubSocket, PubSocket, SocketEventHandle, toggle_fake_events, \
                                set_fake_prefix, get_fake_prefix, delete_fake_prefix, wait_for_one_event
from msgq.ipc_pyx import MultiplePublishersError, IpcError
from msgq import fake_event_handle, pub_sock, sub_sock, drain_sock_raw
import msgq

import os
import capnp
import time
import math

from typing import Optional, List, Union, Dict, Any

from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.util import MovingAverage
from openpilot.common.swaglog import cloudlog

NO_TRAVERSAL_LIMIT = 2**64-1


def reset_context():
  msgq.context = Context()


def log_from_bytes(dat: bytes, struct: capnp.lib.capnp._StructModule = log.Event) -> capnp.lib.capnp._DynamicStructReader:
  with struct.from_bytes(dat, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as msg:
    return msg


def new_message(service: Optional[str], size: Optional[int] = None, **kwargs) -> capnp.lib.capnp._DynamicStructBuilder:
  args = {
    'valid': False,
    'logMonoTime': int(time.monotonic() * 1e9),
    **kwargs
  }
  dat = log.Event.new_message(**args)
  if service is not None:
    if size is None:
      dat.init(service)
    else:
      dat.init(service, size)
  return dat


def drain_sock(sock: SubSocket, wait_for_one: bool = False) -> List[capnp.lib.capnp._DynamicStructReader]:
  """Receive all message currently available on the queue"""
  msgs = drain_sock_raw(sock, wait_for_one=wait_for_one)
  return [log_from_bytes(m) for m in msgs]


# TODO: print when we drop packets?
def recv_sock(sock: SubSocket, wait: bool = False) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  """Same as drain sock, but only returns latest message. Consider using conflate instead."""
  dat = None

  while 1:
    if wait and dat is None:
      recv = sock.receive()
    else:
      recv = sock.receive(non_blocking=True)

    if recv is None:  # Timeout hit
      break

    dat = recv

  if dat is not None:
    dat = log_from_bytes(dat)

  return dat


def recv_one(sock: SubSocket) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  dat = sock.receive()
  if dat is not None:
    dat = log_from_bytes(dat)
  return dat


def recv_one_or_none(sock: SubSocket) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  dat = sock.receive(non_blocking=True)
  if dat is not None:
    dat = log_from_bytes(dat)
  return dat


def recv_one_retry(sock: SubSocket) -> capnp.lib.capnp._DynamicStructReader:
  """Keep receiving until we get a message"""
  while True:
    dat = sock.receive()
    if dat is not None:
      return log_from_bytes(dat)


class FrequencyTracker:
  def __init__(self, service_freq: float, update_freq: float, is_poll: bool):
    freq = max(min(service_freq, update_freq), 1.)
    if is_poll:
      min_freq = max_freq = freq
    else:
      max_freq = min(freq, update_freq)
      if service_freq >= 2 * update_freq:
        min_freq = update_freq
      elif update_freq >= 2* service_freq:
        min_freq = freq
      else:
        min_freq = min(freq, freq / 2.)

    self.min_freq = min_freq * 0.8
    self.max_freq = max_freq * 1.2
    self.avg_dt = MovingAverage(int(10 * freq))
    self.recent_avg_dt = MovingAverage(int(freq))
    self.prev_time = 0.0

  def record_recv_time(self, cur_time: float) -> None:
    # TODO: Handle case where cur_time is less than prev_time
    if self.prev_time > 1e-5:
      dt = cur_time - self.prev_time

      self.avg_dt.add_value(dt)
      self.recent_avg_dt.add_value(dt)

    self.prev_time = cur_time

  @property
  def valid(self) -> bool:
    if self.avg_dt.count == 0:
      return False

    avg_freq = 1.0 / self.avg_dt.get_average()
    if self.min_freq <= avg_freq <= self.max_freq:
      return True

    avg_freq_recent = 1.0 / self.recent_avg_dt.get_average()
    return self.min_freq <= avg_freq_recent <= self.max_freq


class SubMaster:
  def __init__(self, services: List[str], poll: Optional[str] = None,
               ignore_alive: Optional[List[str]] = None, ignore_avg_freq: Optional[List[str]] = None,
               ignore_valid: Optional[List[str]] = None, addr: str = "127.0.0.1", frequency: Optional[float] = None):
    self.frame = -1
    self.services = services
    self.seen = {s: False for s in services}
    self.updated = {s: False for s in services}
    self.recv_time = {s: 0. for s in services}
    self.recv_frame = {s: 0 for s in services}
    self.sock = {}
    self.data = {}
    self.logMonoTime = {s: 0 for s in services}

    # zero-frequency / on-demand services are always alive and presumed valid; all others must pass checks
    on_demand = {s: SERVICE_LIST[s].frequency <= 1e-5 for s in services}
    self.static_freq_services = set(s for s in services if not on_demand[s])
    self.alive = {s: on_demand[s] for s in services}
    self.freq_ok = {s: on_demand[s] for s in services}
    self.valid = {s: on_demand[s] for s in services}

    self.freq_tracker: Dict[str, FrequencyTracker] = {}
    self.poller = Poller()
    polled_services = set([poll, ] if poll is not None else services)
    self.non_polled_services = set(services) - polled_services

    self.ignore_average_freq = [] if ignore_avg_freq is None else ignore_avg_freq
    self.ignore_alive = [] if ignore_alive is None else ignore_alive
    self.ignore_valid = [] if ignore_valid is None else ignore_valid

    self.simulation = bool(int(os.getenv("SIMULATION", "0")))

    # if freq and poll aren't specified, assume the max to be conservative
    assert frequency is None or poll is None, "Do not specify 'frequency' - frequency of the polled service will be used."
    self.update_freq = frequency or max([SERVICE_LIST[s].frequency for s in polled_services])
    self._status_cache: Dict[str, Dict[str, bool]] = {s: {'alive': True, 'freq_ok': True, 'valid': True} for s in services}
    self._startup_health_log_frames = max(5, int(self.update_freq)) if self.update_freq > 0 else 5
    self._last_update_time = 0.0

    for s in services:
      p = self.poller if s not in self.non_polled_services else None
      self.sock[s] = sub_sock(s, poller=p, addr=addr, conflate=True)

      try:
        data = new_message(s)
      except capnp.lib.capnp.KjException:
        data = new_message(s, 0) # lists

      self.data[s] = getattr(data.as_reader(), s)
      self.freq_tracker[s] = FrequencyTracker(SERVICE_LIST[s].frequency, self.update_freq, s == poll)

  def __getitem__(self, s: str) -> capnp.lib.capnp._DynamicStructReader:
    return self.data[s]

  def _check_avg_freq(self, s: str) -> bool:
    return SERVICE_LIST[s].frequency > 0.99 and (s not in self.ignore_average_freq) and (s not in self.ignore_alive)

  def update(self, timeout: int = 100) -> None:
    msgs = []
    for sock in self.poller.poll(timeout):
      msgs.append(recv_one_or_none(sock))

    # non-blocking receive for non-polled sockets
    for s in self.non_polled_services:
      msgs.append(recv_one_or_none(self.sock[s]))
    self.update_msgs(time.monotonic(), msgs)

  def update_msgs(self, cur_time: float, msgs: List[capnp.lib.capnp._DynamicStructReader]) -> None:
    self.frame += 1
    self.updated = dict.fromkeys(self.services, False)
    for msg in msgs:
      if msg is None:
        continue

      s = msg.which()
      self.seen[s] = True
      self.updated[s] = True

      self.freq_tracker[s].record_recv_time(cur_time)
      self.recv_time[s] = cur_time
      self.recv_frame[s] = self.frame
      self.data[s] = getattr(msg, s)
      self.logMonoTime[s] = msg.logMonoTime
      self.valid[s] = msg.valid

    for s in self.static_freq_services:
      # alive if delay is within 10x the expected frequency; checks relaxed in simulator
      self.alive[s] = (cur_time - self.recv_time[s]) < (10. / SERVICE_LIST[s].frequency) or (self.seen[s] and self.simulation)
      self.freq_ok[s] = self.freq_tracker[s].valid or self.simulation

    self._last_update_time = cur_time
    self._log_health_status(cur_time)

  def all_alive(self, service_list: Optional[List[str]] = None) -> bool:
    return all(self.alive[s] for s in (service_list or self.services) if s not in self.ignore_alive)

  def all_freq_ok(self, service_list: Optional[List[str]] = None) -> bool:
    return all(self.freq_ok[s] for s in (service_list or self.services) if self._check_avg_freq(s))

  def all_valid(self, service_list: Optional[List[str]] = None) -> bool:
    return all(self.valid[s] for s in (service_list or self.services) if s not in self.ignore_valid)

  def all_checks(self, service_list: Optional[List[str]] = None) -> bool:
    return self.all_alive(service_list) and self.all_freq_ok(service_list) and self.all_valid(service_list)

  def _service_expected_interval(self, s: str) -> Optional[float]:
    freq = SERVICE_LIST[s].frequency
    return (1.0 / freq) if freq > 0 else None

  def _frequency_stats(self, s: str) -> Dict[str, Optional[float]]:
    tracker = self.freq_tracker[s]
    avg_dt = tracker.avg_dt.get_average()
    recent_dt = tracker.recent_avg_dt.get_average()

    avg_freq = None
    if tracker.avg_dt.count > 0 and math.isfinite(avg_dt) and avg_dt > 0:
      avg_freq = 1.0 / avg_dt

    recent_freq = None
    if tracker.recent_avg_dt.count > 0 and math.isfinite(recent_dt) and recent_dt > 0:
      recent_freq = 1.0 / recent_dt

    return {
      'avg_frequency': avg_freq,
      'recent_frequency': recent_freq,
      'avg_sample_count': tracker.avg_dt.count,
      'recent_sample_count': tracker.recent_avg_dt.count,
    }

  def _service_check_failures(self, s: str) -> Dict[str, bool]:
    return {
      'alive': (not self.alive[s]) and (s not in self.ignore_alive),
      'freq_ok': (not self.freq_ok[s]) and self._check_avg_freq(s),
      'valid': (not self.valid[s]) and (s not in self.ignore_valid),
    }

  def _build_service_health(self, s: str, cur_time: float) -> Dict[str, Any]:
    stats = self._frequency_stats(s)
    last_recv = self.recv_time[s]
    time_since_recv = (cur_time - last_recv) if last_recv > 0 else None
    if time_since_recv is not None:
      time_since_recv = round(time_since_recv, 3)

    avg_freq = stats['avg_frequency']
    if avg_freq is not None:
      avg_freq = round(avg_freq, 3)

    recent_freq = stats['recent_frequency']
    if recent_freq is not None:
      recent_freq = round(recent_freq, 3)

    expected_interval = self._service_expected_interval(s)
    if expected_interval is not None:
      expected_interval = round(expected_interval, 3)

    failures = self._service_check_failures(s)
    effective_status = {k: not failed for k, failed in failures.items()}

    return {
      'service': s,
      'alive': self.alive[s],
      'freq_ok': self.freq_ok[s],
      'valid': self.valid[s],
      'seen': self.seen[s],
      'effectiveStatus': effective_status,
      'checkFailures': failures,
      'failingChecks': [check for check, failed in failures.items() if failed],
      'expectedFrequency': SERVICE_LIST[s].frequency,
      'expectedInterval': expected_interval,
      'timeSinceLastMessage': time_since_recv,
      'avgFrequency': avg_freq,
      'recentFrequency': recent_freq,
      'avgSampleCount': stats['avg_sample_count'],
      'recentSampleCount': stats['recent_sample_count'],
      'recvFrame': self.recv_frame[s],
      'logMonoTime': int(self.logMonoTime[s]),
      'ignoredAliveCheck': s in self.ignore_alive,
      'ignoredFreqCheck': not self._check_avg_freq(s),
      'ignoredValidCheck': s in self.ignore_valid,
    }

  def _emit_service_health_event(self, s: str, cur_time: float, *, issue: bool, changed: bool,
                                 status: Dict[str, bool], previous_status: Optional[Dict[str, bool]]) -> None:
    health = self._build_service_health(s, cur_time)
    health['statusChanged'] = changed
    health['previousStatus'] = previous_status.copy() if isinstance(previous_status, dict) else previous_status
    health['effectiveStatus'] = status.copy()
    health['frame'] = int(self.frame)
    if issue:
      cloudlog.event('submaster_service_issue', error=True, **health)
    else:
      cloudlog.event('submaster_service_recovered', **health)

  def _log_health_status(self, cur_time: float) -> None:
    ready_frame = self._startup_health_log_frames
    for s in self.services:
      failures = self._service_check_failures(s)
      status = {k: not failed for k, failed in failures.items()}
      prev = self._status_cache.get(s)
      changed = prev != status
      issue = any(failures.values())
      ready = self.seen[s] or self.frame >= ready_frame

      if issue and ready and (changed or (not self.seen[s] and self.frame == ready_frame)):
        self._emit_service_health_event(s, cur_time, issue=True, changed=changed, status=status, previous_status=prev)
      elif prev is not None and not issue and not all(prev.values()) and changed:
        self._emit_service_health_event(s, cur_time, issue=False, changed=changed, status=status, previous_status=prev)

      self._status_cache[s] = status.copy()

  def get_health_snapshot(self, service_list: Optional[List[str]] = None, only_issues: bool = False) -> Dict[str, Dict[str, Any]]:
    services = service_list or self.services
    cur_time = self._last_update_time or time.monotonic()
    snapshot: Dict[str, Dict[str, Any]] = {}
    for s in services:
      health = self._build_service_health(s, cur_time)
      if only_issues and not health['failingChecks']:
        continue
      snapshot[s] = health
    return snapshot


class PubMaster:
  def __init__(self, services: List[str]):
    self.sock = {}
    for s in services:
      self.sock[s] = pub_sock(s)

  def send(self, s: str, dat: Union[bytes, capnp.lib.capnp._DynamicStructBuilder]) -> None:
    if not isinstance(dat, bytes):
      dat = dat.to_bytes()
    self.sock[s].send(dat)

  def wait_for_readers_to_update(self, s: str, timeout: int, dt: float = 0.05) -> bool:
    for _ in range(int(timeout*(1./dt))):
      if self.sock[s].all_readers_updated():
        return True
      time.sleep(dt)
    return False

  def all_readers_updated(self, s: str) -> bool:
    return self.sock[s].all_readers_updated()  # type: ignore
