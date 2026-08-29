#!/usr/bin/env python3
"""Bounded, reversible UDS campaign for a Hyundai ADAS DRV ECU.

This tool only emits diagnostic traffic to one physical ECU address. It does not
perform memory access, firmware transfer, RoutineControl, or SecurityAccess key
attempts. Stop openpilot before running it; the panda is placed in ELM327 safety.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from opendbc.car.structs import CarParams
from opendbc.car.uds import (ACCESS_TYPE, CONTROL_TYPE, DTC_SETTING_TYPE, MESSAGE_TYPE, RESET_TYPE,
                             SESSION_TYPE, MessageTimeoutError, NegativeResponseError, UdsClient)
from panda import Panda


DISABLE_CONTROLS = (
  CONTROL_TYPE.ENABLE_RX_DISABLE_TX,
  CONTROL_TYPE.DISABLE_RX_ENABLE_TX,
  CONTROL_TYPE.DISABLE_RX_DISABLE_TX,
)
MESSAGE_TYPES = (
  MESSAGE_TYPE.NORMAL,
  MESSAGE_TYPE.NETWORK_MANAGEMENT,
  MESSAGE_TYPE.NORMAL_AND_NETWORK_MANAGEMENT,
)


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


def describe_exception(exc: Exception) -> dict:
  if isinstance(exc, NegativeResponseError):
    return {"result": "negative_response", "service": exc.service_id, "nrc": exc.error_code, "detail": str(exc)}
  if isinstance(exc, MessageTimeoutError):
    return {"result": "timeout", "detail": str(exc)}
  return {"result": "exception", "type": type(exc).__name__, "detail": str(exc)}


def restore(client: UdsClient, log: JsonLog, message_type: MESSAGE_TYPE):
  for action, fn in (
    ("restore_communications", lambda: client.communication_control(CONTROL_TYPE.ENABLE_RX_ENABLE_TX, message_type)),
    ("restore_dtc", lambda: client.control_dtc_setting(DTC_SETTING_TYPE.ON)),
    ("restore_default_session", lambda: client.diagnostic_session_control(SESSION_TYPE.DEFAULT)),
  ):
    try:
      fn()
      log.write(action, result="positive")
    except Exception as exc:
      log.write(action, **describe_exception(exc))


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
  parser.add_argument("--timeout", type=float, default=0.5)
  parser.add_argument("--max-attempts", type=int, default=108)
  parser.add_argument("--max-runtime", type=float, default=180.0)
  parser.add_argument("--hold-seconds", type=float, default=15.0,
                      help="on success, maintain Tester Present for this long before restoring")
  parser.add_argument("--include-programming-session", action="store_true")
  parser.add_argument("--include-safety-session", action="store_true")
  parser.add_argument("--include-soft-reset", action="store_true")
  parser.add_argument("--probe-security-seeds", action="store_true",
                      help="request one seed at levels 1, 3, and 5; never computes or sends keys")
  parser.add_argument("--output", type=Path,
                      default=Path(f"/tmp/hyundai-adas-campaign-{int(time.time())}.jsonl"))
  args = parser.parse_args()

  if not 0 <= args.bus <= 3 or not 0 < args.timeout <= 2 or args.max_attempts <= 0 or args.max_runtime <= 0:
    parser.error("invalid campaign bounds")

  serials = Panda.list()
  if not serials:
    raise SystemExit("no panda found")
  if args.serial is None and len(serials) != 1:
    raise SystemExit(f"found {len(serials)} pandas; pass --serial")

  log = JsonLog(args.output)
  panda = Panda(serial=args.serial)
  started = time.monotonic()
  success = None
  try:
    # ELM327 safety permits diagnostic addressing but blocks normal actuation frames.
    panda.set_safety_mode(CarParams.SafetyModel.elm327, 1)
    client = UdsClient(panda, args.addr, bus=args.bus, timeout=args.timeout, response_pending_timeout=2.0)
    log.write("campaign_start", addr=args.addr, bus=args.bus, serial=panda.get_serial()[0],
              max_attempts=args.max_attempts, max_runtime=args.max_runtime)

    client.tester_present()
    log.write("baseline_tester_present", result="positive")

    if args.probe_security_seeds:
      client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
      for level in (1, 3, 5):
        try:
          seed = client.security_access(ACCESS_TYPE(level))
          log.write("security_seed", level=level, result="positive", seed_length=len(seed))
        except Exception as exc:
          log.write("security_seed", level=level, **describe_exception(exc))
      restore(client, log, MESSAGE_TYPE.NORMAL)

    attempts = build_attempts(args.include_programming_session, args.include_safety_session, args.include_soft_reset)
    for index, attempt in enumerate(attempts[:args.max_attempts], 1):
      if time.monotonic() - started > args.max_runtime:
        log.write("circuit_breaker", reason="runtime", attempt=index)
        break
      log.write("attempt", index=index, **attempt.__dict__)
      try:
        client.diagnostic_session_control(SESSION_TYPE.DEFAULT)
        if attempt.soft_reset:
          client.ecu_reset(RESET_TYPE.SOFT)
          time.sleep(0.5)
        client.diagnostic_session_control(attempt.session)
        if attempt.tester_before:
          client.tester_present()
        if attempt.dtc_off:
          client.control_dtc_setting(DTC_SETTING_TYPE.OFF)
        time.sleep(attempt.settle_s)
        client.communication_control(attempt.control, attempt.message_type)
        success = attempt
        log.write("success", index=index, **attempt.__dict__)
        print(
          f"\nSUCCESS: attempt {index}: {attempt.name}\n"
          f"  session={attempt.session.name}\n"
          f"  control={attempt.control.name}\n"
          f"  message_type={attempt.message_type.name}\n",
          flush=True,
        )
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
          time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
          if time.monotonic() < deadline:
            client.tester_present()
        restore(client, log, attempt.message_type)
        break
      except Exception as exc:
        log.write("attempt_result", index=index, **describe_exception(exc))
        restore(client, log, attempt.message_type)
        time.sleep(0.1)

    log.write("campaign_end", success=success is not None,
              successful_attempt=success.__dict__ if success is not None else None,
              elapsed=round(time.monotonic() - started, 3))
    if success is None:
      print("\nNO SUCCESS: every permitted campaign attempt failed or the campaign limit was reached.\n", flush=True)
    return 0 if success is not None else 2
  finally:
    try:
      panda.set_safety_mode(CarParams.SafetyModel.noOutput)
      panda.close()
    finally:
      log.close()


if __name__ == "__main__":
  raise SystemExit(main())
