#!/usr/bin/env python3
"""Bounded, reversible UDS campaign for a Hyundai ADAS DRV ECU.

This tool only emits diagnostic traffic to one physical ECU address. It does not
perform memory access, firmware transfer, RoutineControl, or SecurityAccess key
attempts. Stop openpilot before running it; the panda is placed in ELM327 safety.
"""

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

from opendbc.car.carlog import carlog
from opendbc.car.structs import CarParams
from opendbc.car.uds import (ACCESS_TYPE, CONTROL_TYPE, DTC_SETTING_TYPE, MESSAGE_TYPE, RESET_TYPE,
                             SESSION_TYPE, MessageTimeoutError, NegativeResponseError, UdsClient)
from panda import Panda


DISABLE_CONTROLS = (
  CONTROL_TYPE.ENABLE_RX_DISABLE_TX,
)
MESSAGE_TYPES = (
  MESSAGE_TYPE.NORMAL,
)
DEFAULT_VERIFY_ADDRS = (0x12A, 0x1A0)  # LFA and SCC_CONTROL, both transmitted by ADAS_DRV on ECAN


@dataclass(frozen=True)
class Attempt:
  name: str
  session: SESSION_TYPE
  control: CONTROL_TYPE
  message_type: MESSAGE_TYPE
  dtc_off: bool = False
  soft_reset: bool = False
  tester_before: bool = False
  settle_s: float = 0.05


@dataclass(frozen=True)
class CanCapture:
  counts: dict[int, int]
  total: int


class JsonLog:
  def __init__(self, path: Path):
    self.path = path
    self.f = path.open("a", encoding="utf-8")

  def write(self, event: str, **data):
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    self.f.write(line + "\n")
    self.f.flush()

  def close(self):
    self.f.close()


class BoundedCanReceiver:
  """Prevent UdsClient's pre-request drain from looping forever on a busy CAN-FD bus.

  CanClient keeps draining while a receive call returns exactly 254 frames. On a
  busy vehicle that condition can remain true indefinitely. Returning at most
  253 frames preserves every frame in a local FIFO while guaranteeing each drain
  call terminates.
  """

  def __init__(self, recv: Callable[[], list[tuple[int, bytes, int]]], max_batch: int = 253):
    self.recv = recv
    self.max_batch = max_batch
    self.pending: deque[tuple[int, bytes, int]] = deque()

  def __call__(self) -> list[tuple[int, bytes, int]]:
    if not self.pending:
      self.pending.extend(self.recv() or [])
    count = min(self.max_batch, len(self.pending))
    return [self.pending.popleft() for _ in range(count)]

  def clear(self):
    self.pending.clear()


def describe_exception(exc: Exception) -> dict:
  if isinstance(exc, NegativeResponseError):
    return {"result": "negative_response", "service": exc.service_id, "nrc": exc.error_code, "detail": str(exc)}
  if isinstance(exc, MessageTimeoutError):
    return {"result": "timeout", "detail": str(exc)}
  return {"result": "exception", "type": type(exc).__name__, "detail": str(exc)}


def uds_action(log: JsonLog, action: str, request: bytes, fn: Callable[[], object], **data):
  started = time.monotonic()
  log.write("uds_request", action=action, request=request.hex(), **data)
  try:
    result = fn()
  except Exception as exc:
    log.write("uds_response", action=action, request=request.hex(), elapsed=round(time.monotonic() - started, 3),
              **data, **describe_exception(exc))
    raise
  log.write("uds_response", action=action, request=request.hex(), result="positive",
            elapsed=round(time.monotonic() - started, 3), **data)
  return result


def raw_uds_single_frame(panda: Panda, log: JsonLog, addr: int, bus: int, payload: bytes, action: str):
  if not 1 <= len(payload) <= 7:
    raise ValueError("raw single-frame UDS payload must be 1..7 bytes")
  frame = (bytes([len(payload)]) + payload).ljust(8, b"\x00")
  log.write("uds_suppressed_request", action=action, addr=addr, bus=bus,
            request=payload.hex(), can_data=frame.hex())
  panda.can_send(addr, frame, bus)


def restore_suppressed(panda: Panda, log: JsonLog, addr: int, bus: int, message_type: MESSAGE_TYPE,
                       dtc_was_disabled: bool):
  """Restore state without waiting for replies that can be delayed and mis-associated."""
  raw_uds_single_frame(panda, log, addr, bus,
                       bytes([0x28, 0x80 | CONTROL_TYPE.ENABLE_RX_ENABLE_TX, message_type]),
                       "restore_communications_suppress_response")
  time.sleep(0.1)
  if dtc_was_disabled:
    raw_uds_single_frame(panda, log, addr, bus, bytes([0x85, 0x80 | DTC_SETTING_TYPE.ON]),
                         "restore_dtc_suppress_response")
    time.sleep(0.1)
  raw_uds_single_frame(panda, log, addr, bus, bytes([0x10, 0x80 | SESSION_TYPE.DEFAULT]),
                       "restore_default_session_suppress_response")


def clear_rx(panda: Panda, receiver: BoundedCanReceiver, log: JsonLog, reason: str):
  receiver.clear()
  panda.can_clear(0xFFFF)
  log.write("can_rx_cleared", reason=reason)


def capture_can(receiver: BoundedCanReceiver, log: JsonLog, phase: str, bus: int,
                watched_addrs: tuple[int, ...], duration: float) -> CanCapture:
  counts: Counter[int] = Counter()
  total = 0
  started = time.monotonic()
  deadline = started + duration
  log.write("can_capture_start", phase=phase, bus=bus, duration=duration,
            watched_addrs=[hex(addr) for addr in watched_addrs])
  while time.monotonic() < deadline:
    for addr, _dat, rx_bus in receiver():
      if rx_bus != bus:
        continue
      total += 1
      if addr in watched_addrs:
        counts[addr] += 1
  result = {addr: counts[addr] for addr in watched_addrs}
  log.write("can_capture_end", phase=phase, bus=bus, elapsed=round(time.monotonic() - started, 3), total=total,
            watched_counts={hex(addr): count for addr, count in result.items()})
  return CanCapture(result, total)


def verify_silence(log: JsonLog, baseline: CanCapture, after: CanCapture, min_baseline: int,
                   min_total: int, phase: str) -> tuple[bool, list[int]]:
  observed = [addr for addr, count in baseline.counts.items() if count >= min_baseline]
  still_transmitting = [addr for addr in observed if after.counts.get(addr, 0) != 0]
  bus_alive = after.total >= min_total
  verified = bool(observed) and bus_alive and not still_transmitting
  log.write("disable_verification", phase=phase, result="verified_silent" if verified else "not_verified",
            min_baseline=min_baseline, observed=[hex(addr) for addr in observed],
            still_transmitting=[hex(addr) for addr in still_transmitting], bus_alive=bus_alive,
            post_disable_bus_frames=after.total, min_bus_frames=min_total)
  return verified, observed


def verify_recovery(log: JsonLog, baseline: CanCapture, after_restore: CanCapture, observed: list[int],
                    min_baseline: int) -> bool:
  recovered = [addr for addr in observed if after_restore.counts.get(addr, 0) >= min_baseline]
  verified = bool(observed) and len(recovered) == len(observed)
  log.write("restore_verification", result="verified_recovered" if verified else "not_verified",
            expected=[hex(addr) for addr in observed], recovered=[hex(addr) for addr in recovered],
            baseline={hex(addr): count for addr, count in baseline.counts.items()},
            after_restore={hex(addr): count for addr, count in after_restore.counts.items()})
  return verified


def build_attempts(include_programming: bool, include_safety_session: bool, include_reset: bool) -> list[Attempt]:
  attempts = []
  sessions = [SESSION_TYPE.EXTENDED_DIAGNOSTIC]
  if include_safety_session:
    sessions.append(SESSION_TYPE.SAFETY_SYSTEM_DIAGNOSTIC)
  if include_programming:
    sessions.append(SESSION_TYPE.PROGRAMMING)

  for session in sessions:
    for control in DISABLE_CONTROLS:
      for message_type in MESSAGE_TYPES:
        base = f"{session.name.lower()}-{control.name.lower()}-{message_type.name.lower()}"
        attempts.append(Attempt(base, session, control, message_type))
        attempts.append(Attempt(base + "-tester-first", session, control, message_type, tester_before=True, settle_s=0.2))
        attempts.append(Attempt(base + "-dtc-off", session, control, message_type, dtc_off=True, tester_before=True, settle_s=0.2))
        if include_reset:
          attempts.append(Attempt(base + "-soft-reset", session, control, message_type, soft_reset=True, tester_before=True, settle_s=0.1))
  return attempts


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", help="panda serial; required when more than one panda is connected")
  parser.add_argument("--bus", type=int, default=1)
  parser.add_argument("--addr", type=lambda x: int(x, 0), default=0x730)
  parser.add_argument("--timeout", type=float, default=2.0)
  parser.add_argument("--max-attempts", type=int, default=108)
  parser.add_argument("--max-runtime", type=float, default=180.0)
  parser.add_argument("--hold-seconds", type=float, default=15.0,
                      help="on success, maintain Tester Present for this long before restoring")
  parser.add_argument("--verify-addr", action="append", type=lambda x: int(x, 0), dest="verify_addrs",
                      help="ECU-owned cyclic CAN address to monitor; repeat for multiple (default: 0x12a and 0x1a0)")
  parser.add_argument("--verify-window", type=float, default=1.5)
  parser.add_argument("--verify-min-baseline", type=int, default=3)
  parser.add_argument("--verify-min-bus-frames", type=int, default=10,
                      help="minimum non-target bus traffic required after disable to rule out a dead/disconnected bus")
  parser.add_argument("--include-programming-session", action="store_true")
  parser.add_argument("--include-safety-session", action="store_true")
  parser.add_argument("--include-soft-reset", action="store_true")
  parser.add_argument("--probe-security-seeds", action="store_true",
                      help="request one seed at levels 1, 3, and 5; never computes or sends keys")
  parser.add_argument("--output", type=Path,
                      default=Path(f"/tmp/hyundai-adas-campaign-{int(time.time())}.jsonl"))
  args = parser.parse_args()

  if not 0 <= args.bus <= 3 or not 0 < args.timeout <= 5 or args.max_attempts <= 0 or args.max_runtime <= 0 or \
     not 0.1 <= args.verify_window <= 10 or args.verify_min_baseline <= 0 or args.verify_min_bus_frames <= 0:
    parser.error("invalid campaign bounds")
  verify_addrs = tuple(dict.fromkeys(args.verify_addrs or DEFAULT_VERIFY_ADDRS))

  serials = Panda.list()
  if not serials:
    raise SystemExit("no panda found")
  if args.serial is None and len(serials) != 1:
    raise SystemExit(f"found {len(serials)} pandas; pass --serial")

  log = JsonLog(args.output)
  panda = Panda(serial=args.serial)
  receiver = BoundedCanReceiver(panda.can_recv)
  started = time.monotonic()
  success = None
  final_status = "FAILED"
  try:
    carlog.setLevel("DEBUG")
    # ELM327 safety permits diagnostic addressing but blocks normal actuation frames.
    panda.set_safety_mode(CarParams.SafetyModel.elm327, 1)
    client = UdsClient(panda, args.addr, bus=args.bus, timeout=args.timeout, response_pending_timeout=5.0)
    # UdsClient stores the bound callback, so replace it after construction.
    client._can_client.rx = receiver
    log.write("campaign_start", addr=args.addr, bus=args.bus, serial=panda.get_serial()[0],
              max_attempts=args.max_attempts, max_runtime=args.max_runtime, uds_timeout=args.timeout,
              verify_addrs=[hex(addr) for addr in verify_addrs], verify_window=args.verify_window,
              verify_min_baseline=args.verify_min_baseline, verify_min_bus_frames=args.verify_min_bus_frames)

    clear_rx(panda, receiver, log, "before_baseline_capture")
    baseline = capture_can(receiver, log, "baseline", args.bus, verify_addrs, args.verify_window)
    if not any(count >= args.verify_min_baseline for count in baseline.counts.values()):
      log.write("campaign_end", success=False, status="INCONCLUSIVE_NO_BASELINE_TRAFFIC",
                baseline={hex(addr): count for addr, count in baseline.counts.items()}, baseline_bus_frames=baseline.total,
                elapsed=round(time.monotonic() - started, 3))
      print("\nINCONCLUSIVE: no monitored ADAS_DRV traffic was present before the test. ECU disable cannot be verified.\n",
            flush=True)
      return 3

    clear_rx(panda, receiver, log, "before_baseline_tester_present")
    uds_action(log, "baseline_tester_present", b"\x3e\x00", client.tester_present)

    if args.probe_security_seeds:
      uds_action(log, "security_extended_session", bytes([0x10, SESSION_TYPE.EXTENDED_DIAGNOSTIC]),
                 lambda: client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC))
      for level in (1, 3, 5):
        try:
          seed = uds_action(log, "security_seed", bytes([0x27, level]),
                            lambda level=level: client.security_access(ACCESS_TYPE(level)), level=level)
          log.write("security_seed", level=level, result="positive", seed_length=len(seed))
        except Exception as exc:
          log.write("security_seed", level=level, **describe_exception(exc))
      restore_suppressed(panda, log, args.addr, args.bus, MESSAGE_TYPE.NORMAL, True)
      time.sleep(0.5)
      clear_rx(panda, receiver, log, "after_security_probe_restore")

    attempts = build_attempts(args.include_programming_session, args.include_safety_session, args.include_soft_reset)
    for index, attempt in enumerate(attempts[:args.max_attempts], 1):
      if time.monotonic() - started > args.max_runtime:
        log.write("circuit_breaker", reason="runtime", attempt=index)
        break
      request = bytes([0x28, attempt.control, attempt.message_type])
      log.write("attempt", index=index, name=attempt.name, session=attempt.session.name, control=attempt.control.name,
                message_type=attempt.message_type.name, request=request.hex(), dtc_off=attempt.dtc_off,
                soft_reset=attempt.soft_reset, tester_before=attempt.tester_before, settle_s=attempt.settle_s)
      try:
        uds_action(log, "default_session", bytes([0x10, SESSION_TYPE.DEFAULT]),
                   lambda: client.diagnostic_session_control(SESSION_TYPE.DEFAULT), attempt=index)
        if attempt.soft_reset:
          uds_action(log, "soft_reset", bytes([0x11, RESET_TYPE.SOFT]),
                     lambda: client.ecu_reset(RESET_TYPE.SOFT), attempt=index)
          time.sleep(0.5)
        uds_action(log, "diagnostic_session", bytes([0x10, attempt.session]),
                   lambda: client.diagnostic_session_control(attempt.session), attempt=index,
                   session=attempt.session.name)
        if attempt.tester_before:
          uds_action(log, "tester_present_before", b"\x3e\x00", client.tester_present, attempt=index)
        if attempt.dtc_off:
          uds_action(log, "dtc_off", bytes([0x85, DTC_SETTING_TYPE.OFF]),
                     lambda: client.control_dtc_setting(DTC_SETTING_TYPE.OFF), attempt=index)
        time.sleep(attempt.settle_s)
        communication_response = "positive"
        try:
          uds_action(log, "communication_control", request,
                     lambda: client.communication_control(attempt.control, attempt.message_type), attempt=index,
                     control=attempt.control.name, message_type=attempt.message_type.name)
          log.write("communication_control_positive", index=index, name=attempt.name, request=request.hex())
        except MessageTimeoutError:
          # Some ECUs apply DisableTx before emitting the positive response. A
          # timeout is therefore ambiguous; cyclic traffic is the authority.
          communication_response = "no_response"
          log.write("communication_control_ambiguous", index=index, name=attempt.name, request=request.hex(),
                    reason="no_response_verify_can_traffic")

        clear_rx(panda, receiver, log, "before_post_disable_capture")
        after_disable = capture_can(receiver, log, "post_disable", args.bus, verify_addrs, args.verify_window)
        disabled, observed = verify_silence(log, baseline, after_disable, args.verify_min_baseline,
                                            args.verify_min_bus_frames, "post_disable")
        if not disabled:
          log.write("attempt_result", index=index, result="not_disabled", communication_response=communication_response)
          restore_suppressed(panda, log, args.addr, args.bus, attempt.message_type, attempt.dtc_off)
          time.sleep(0.5)
          clear_rx(panda, receiver, log, "after_failed_attempt_restore")
          client = UdsClient(panda, args.addr, bus=args.bus, timeout=args.timeout, response_pending_timeout=5.0)
          client._can_client.rx = receiver
          continue

        success = attempt
        final_status = "VERIFIED_DISABLED"
        log.write("success", index=index, name=attempt.name, status=final_status, request=request.hex())
        print(
          f"\nVERIFIED DISABLED: attempt {index}: {attempt.name}\n"
          f"  session={attempt.session.name}\n"
          f"  control={attempt.control.name}\n"
          f"  message_type={attempt.message_type.name}\n",
          flush=True,
        )
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
          time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
          if time.monotonic() < deadline:
            raw_uds_single_frame(panda, log, args.addr, args.bus, b"\x3e\x80",
                                 "hold_tester_present_suppress_response")

        clear_rx(panda, receiver, log, "before_end_hold_capture")
        end_hold = capture_can(receiver, log, "end_hold", args.bus, verify_addrs, args.verify_window)
        held_disabled, _ = verify_silence(log, baseline, end_hold, args.verify_min_baseline,
                                          args.verify_min_bus_frames, "end_hold")
        if not held_disabled:
          final_status = "DISABLE_DID_NOT_HOLD"

        restore_suppressed(panda, log, args.addr, args.bus, attempt.message_type, attempt.dtc_off)
        time.sleep(0.5)
        clear_rx(panda, receiver, log, "before_post_restore_capture")
        after_restore = capture_can(receiver, log, "post_restore", args.bus, verify_addrs, args.verify_window)
        recovery_verified = verify_recovery(log, baseline, after_restore, observed, args.verify_min_baseline)
        if held_disabled and recovery_verified:
          final_status = "VERIFIED_DISABLED_AND_RESTORED"
        elif held_disabled:
          final_status = "VERIFIED_DISABLED_RESTORE_UNVERIFIED"
        break
      except Exception as exc:
        log.write("attempt_result", index=index, **describe_exception(exc))
        restore_suppressed(panda, log, args.addr, args.bus, attempt.message_type, attempt.dtc_off)
        if success is attempt:
          final_status = "VERIFIED_DISABLED_TEST_ERROR"
          break
        time.sleep(0.1)

    log.write("campaign_end", success=success is not None, status=final_status,
              successful_attempt=success.name if success is not None else None,
              elapsed=round(time.monotonic() - started, 3))
    if success is None:
      print("\nNO SUCCESS: every permitted campaign attempt failed or the campaign limit was reached.\n", flush=True)
    else:
      print(f"\nFINAL STATUS: {final_status}\n", flush=True)
    return 0 if final_status == "VERIFIED_DISABLED_AND_RESTORED" else (4 if success is not None else 2)
  except Exception as exc:
    log.write("campaign_end", success=False, status="UNEXPECTED_ERROR",
              elapsed=round(time.monotonic() - started, 3), **describe_exception(exc))
    print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}\n", flush=True)
    return 5
  finally:
    try:
      panda.set_safety_mode(CarParams.SafetyModel.noOutput)
      panda.close()
    finally:
      log.close()


if __name__ == "__main__":
  raise SystemExit(main())
