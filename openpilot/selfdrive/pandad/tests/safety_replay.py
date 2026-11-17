#!/usr/bin/env python3
# checks the panda's safety against libsafety on the car test routes
import hashlib
import multiprocessing
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from opendbc.car.tests.routes import routes
from opendbc.car.tests.test_models import NUM_JOBS, JOB_ID
from opendbc.safety.tests.libsafety import libsafety_py
from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware import HARDWARE
from openpilot.selfdrive.pandad.tests.safety_diff import BUSES, DT, load, replay

SECONDS_PER_ROUTE = 10
# seconds until the RX checks are compared
STEADY = 2
# master's passing runs, restored with the alfred cache
PASSED = Path("/data/scons_cache/safety_replay")
# a pass is reused for a day, so the rack itself is still checked daily
PASS_TTL = 24 * 3600
BUILD_OUTPUTS = {".so", ".o", ".os", ".a", ".pyc", ".gcno", ".gcda"}


def inputs() -> str:
  # everything the result depends on, except the rack: the firmware, opendbc, the panda library, this test, the OS and its compiler
  root = Path(BASEDIR)
  files = sorted((root / "panda/board/obj").glob("*.bin.signed"))
  for d in ("opendbc_repo/opendbc", "panda/python"):
    files += sorted(f for f in (root / d).rglob("*") if f.is_file() and f.suffix not in BUILD_OUTPUTS and "__pycache__" not in f.parts)
  files += [Path(__file__).resolve(), Path(__file__).with_name("safety_diff.py").resolve(), root / "openpilot/selfdrive/pandad/pandad.py"]
  h = hashlib.sha256()
  for f in files:
    h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes())
  h.update(f"{HARDWARE.get_os_version()}".encode() + subprocess.check_output(["cc", "--version"]))
  h.update(f"{NUM_JOBS} {JOB_ID}".encode())
  return h.hexdigest()[:16]


def passed_age(key: str) -> float | None:
  try:
    # wall clock, the pass was written on another DUT
    age = time.time() - float((PASSED / key).read_text())  # noqa: TID251
  except (OSError, ValueError):
    return None
  return age if 0 <= age < PASS_TTL else None


def load_window(route):
  # a route download can drop mid file
  for attempt in range(3):
    try:
      return load(route, SECONDS_PER_ROUTE)
    except Exception:
      if attempt == 2:
        raise
      time.sleep(10)


def without_forwarded_copies(mode, param, frames):
  # drop the panda's forwarded copies in the log, the panda forwards them again
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(mode, param) == 0
  out = []
  for frame in frames:
    copies = Counter(frame)
    drop = Counter()
    for (addr, dat, bus), n in copies.items():
      if bus == 0 and (addr, dat, 2) in copies and safety.safety_fwd_hook(0, addr) == 2:
        drop[(addr, dat, 2)] += min(n, copies[(addr, dat, 2)])
      elif bus == 2 and (addr, dat, 0) in copies and safety.safety_fwd_hook(2, addr) == 0 and safety.safety_fwd_hook(0, addr) != 2:
        drop[(addr, dat, 0)] += min(n, copies[(addr, dat, 0)])
    kept = []
    for msg in frame:
      if drop[msg]:
        drop[msg] -= 1
      else:
        kept.append(msg)
    out.append(kept)
  return out


def expected(mode, param, frames):
  checks_invalid = []

  def on_tick(i, safety):
    checks_invalid.append(i * DT >= STEADY and bool(safety.safety_a.state.safety_rx_checks_invalid))

  steps = replay(mode, param, frames, on_tick)
  rx, fwd, invalid = Counter(), Counter(), 0
  for frame, (results, _) in zip(frames, steps, strict=True):
    for (_, _, bus), (valid, destination) in zip(frame, results, strict=True):
      rx[bus] += 1
      fwd[bus] += destination >= 0
      invalid += not valid
  return {"rx": [rx[b] for b in range(BUSES)], "fwd": [fwd[b] for b in range(BUSES)], "rx_invalid": invalid,
          "relay_malfunction": bool(libsafety_py.libsafety.get_relay_malfunction()), "rx_checks_invalid": any(checks_invalid),
          "safety": [mode, param], "faults": 0, "tx_buffer_overflow": 0, "heartbeat_disabled": False, "harness_changed": False}


def prepare(route):
  mode, param, frames = load_window(route)
  frames = without_forwarded_copies(mode, param, frames)
  # the UP sends a 10 ms step's frames on different buses in any order
  lo, hi = (expected(mode, param, [sorted(f, key=lambda m: m[2] * order) for f in frames]) for order in (1, -1))
  return mode, param, frames, lo, hi


def new_pool():
  return ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))


def within(lo, hi, got):
  if isinstance(lo, list):
    return all(within(a, b, g) for a, b, g in zip(lo, hi, got, strict=True))
  return min(lo, hi) <= got <= max(lo, hi)


def played(p, up, mode, param, frames):
  # first, it resets the heartbeat timeout that turns power save back on
  p.set_safety_mode(mode, param)
  # the panda went into power save while the route loaded without a heartbeat
  p.set_power_save(0)
  # can_health is per CAN controller, a flipped harness swaps buses 0 and 2
  cans = [2, 1, 0] if p.health()["car_harness_status"] == p.HARNESS_STATUS_FLIPPED else [0, 1, 2]
  hs0, h0 = p.health(), [p.can_health(c) for c in cans]
  fd = {bus for frame in frames for _, dat, bus in frame if len(dat) > 8}
  # the panda forwards CAN FD frames between buses 0 and 2
  if fd & {0, 2}:
    fd |= {0, 2}
  for b in fd:
    up.set_can_data_speed_kbps(b, 2000)
  # pad the end so the UP doesn't loop the route
  up.can_replay(frames + [[]] * int(60 / DT))
  start, checks_invalid = time.monotonic(), False
  while time.monotonic() < start + len(frames) * DT + 1:
    p.send_heartbeat(engaged=False)
    # the panda updates its RX checks once a second
    if STEADY + 1 <= time.monotonic() - start <= len(frames) * DT:
      checks_invalid |= p.health()["safety_rx_checks_invalid"]
    time.sleep(0.2)
  h, h1 = p.health(), [p.can_health(c) for c in cans]
  up.reset()
  up.set_ignition(True)
  return {"rx": [h1[b]["total_rx_cnt"] - h0[b]["total_rx_cnt"] for b in range(BUSES)],
          "fwd": [h1[b]["total_fwd_cnt"] - h0[b]["total_fwd_cnt"] for b in range(BUSES)],
          "rx_invalid": h["safety_rx_invalid"], "relay_malfunction": bool(h["faults"] & 1), "rx_checks_invalid": checks_invalid,
          "safety": [h["safety_mode"], h["safety_param"]],
          "faults": h["faults"] & ~hs0["faults"] & ~1,
          # TX only, RX overflows without pandad
          "tx_buffer_overflow": h["tx_buffer_overflow"] - hs0["tx_buffer_overflow"],
          "heartbeat_disabled": h["heartbeat_disabled"], "harness_changed": h["car_harness_status"] != hs0["car_harness_status"],
          # only printed, to debug a failure
          "can_errors": {k: [h1[b][k] - h0[b][k] for b in range(BUSES)]
                         for k in ("total_error_cnt", "total_tx_lost_cnt", "bus_off_cnt", "can_core_reset_count")}}


def main():
  from panda import Panda
  from ultimate_provisioning import UP
  from openpilot.selfdrive.pandad.pandad import flash_panda

  key = inputs()
  age = passed_age(key)
  if age is None:
    print(f"inputs {key} have no master pass in the last {PASS_TTL // 3600} h, replaying", flush=True)
  elif os.environ.get("SAFETY_REPLAY_FORCE") == "1":
    print(f"inputs {key} passed on master {age / 3600:.1f} h ago, replaying anyway (SAFETY_REPLAY_FORCE)", flush=True)
  else:
    print(f"skipped: master passed these inputs ({key}) {age / 3600:.1f} h ago, same firmware, opendbc, panda library, test and OS")
    return 0

  flash_panda()
  os.environ["SAFETY_UBSAN_TRAP"] = "1"
  p, up = Panda(), UP.remote()
  # the UP starts with the harness lines floating
  up.reset()
  up.set_ignition(True)

  names = sys.argv[1:]
  todo = [r for i, r in enumerate(routes) if i % NUM_JOBS == JOB_ID and (not names or r.route in names)]
  failed = []
  # prepares the next route while this one replays
  pool = new_pool()
  loading = pool.submit(prepare, todo[0]) if todo else None
  for n, route in enumerate(todo):
    st = time.monotonic()
    try:
      prepared = loading.result()
    except BrokenProcessPool as e:
      # a crash in libsafety, like a UBSan trap
      prepared = e
      pool = new_pool()
    except Exception as e:
      prepared = e
    if n + 1 < len(todo):
      loading = pool.submit(prepare, todo[n + 1])
    if isinstance(prepared, Exception):
      print(f"{route.route} FAIL, can't prepare: {type(prepared).__name__}: {prepared!r}", flush=True)
      failed.append(route.route)
      continue
    mode, param, frames, lo, hi = prepared
    got = played(p, up, mode, param, frames)
    # forwarding stops at a relay malfunction, exactly when depends on the tick
    if lo["relay_malfunction"] or hi["relay_malfunction"]:
      for d in (lo, hi, got):
        del d["fwd"]
    ok = all(within(lo[k], hi[k], got[k]) for k in lo)
    exp = lo if lo == hi else {"low": lo, "high": hi}
    dropped = [e - g for e, g in zip(lo["rx"], got["rx"], strict=True)]
    print(f"{route.route} {route.car_model} mode {mode} {'ok' if ok else 'FAIL'} {time.monotonic() - st:.0f}s expected {exp} panda {got}" +
          (f" UP dropped {dropped}" if any(dropped) else ""), flush=True)
    if not ok:
      failed.append(route.route)

  pool.shutdown()
  print(f"{len(failed)} failed: {failed}")
  if failed or names:
    return int(bool(failed))
  PASSED.mkdir(parents=True, exist_ok=True)
  (PASSED / key).write_text(str(time.time()))  # noqa: TID251
  return 0


if __name__ == "__main__":
  sys.exit(main())
