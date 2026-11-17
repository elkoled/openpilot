#!/usr/bin/env python3
# compares every safety RX and TX result and the safety state on the car test routes, between BASE's opendbc and this one
#   safety_diff.py [BASE]    BASE defaults to HEAD^, which is master on a PR's merge commit
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DT = 0.01
BUSES = 3
# a TX message's bus, like cereal's can src
TX = 128
STATE = ("get_controls_allowed", "get_relay_malfunction", "get_gas_pressed_prev", "get_brake_pressed_prev", "get_regen_braking_prev",
         "get_steering_disengage_prev", "get_cruise_engaged_prev", "get_vehicle_moving", "get_acc_main_on", "get_torque_meas_min",
         "get_torque_meas_max", "get_torque_driver_min", "get_torque_driver_max", "get_angle_meas_min", "get_angle_meas_max",
         "get_curvature_meas_min", "get_curvature_meas_max")


def load(route, seconds: float | None = None, tx: bool = False):
  from opendbc.car.car_helpers import interfaces
  from opendbc.car.tests.test_models import TestCarModelBase

  R = type("R", (TestCarModelBase,), {"test_route": route, "platform": route.car_model})
  car_fw, can_msgs, alpha_long = R.get_testing_data(route)
  CP = interfaces[R.platform].get_params(R.platform, R.fingerprint, car_fw, alpha_long, False, docs=False)
  cfg = CP.safetyConfigs[-1]

  t0 = can_msgs[0][0]
  steps = int((can_msgs[-1][0] - t0) / 1e9 / DT) + 1 if seconds is None else int(seconds / DT)
  frames = [[] for _ in range(steps)]
  for t, msgs in can_msgs:
    i = int((t - t0) / 1e9 / DT)
    if i >= len(frames):
      break
    for m in msgs:
      if m.src < 64 and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), m.src % 4))
      elif tx and TX <= m.src < TX + 64 and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), TX + m.src % 4))
  return cfg.safetyModel.raw, cfg.safetyParam, frames


def replay(mode: int, param: int, frames, on_tick=None):
  # the result of each message and the safety state after each 10 ms step
  from opendbc.safety.tests.libsafety import libsafety_py
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(mode, param) == 0
  safety.set_alternative_experience(0)

  steps = []
  for i, frame in enumerate(frames):
    safety.set_timer(int(i * DT * 1e6))
    if i % int(1 / DT) == 0:
      safety.safety_tick()
      if on_tick is not None:
        on_tick(i, safety)
    results = []
    for addr, dat, bus in frame:
      if bus >= TX:
        results.append(bool(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus - TX, dat))))
      else:
        fwd = safety.safety_fwd_hook(bus, addr)
        results.append((bool(safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, bus, dat))), fwd))
    steps.append((results, tuple(getattr(safety, g)() for g in STATE)))
  return steps


def dump_route(args):
  i, route, out = args
  try:
    with open(out / f"{i}.pkl", "wb") as f:
      pickle.dump((route.route, route.car_model, *load(route, tx=True)), f)
  except Exception as e:
    return f"{route.route} can't load: {e!r}"
  return None


def digest_route(path):
  # per second: a hash of everything, then what to show if it differs: valid RX, allowed TX and each state value, summed
  with open(path, "rb") as f:
    route, model, mode, param, frames = pickle.load(f)
  steps = replay(mode, param, frames)
  seconds = []
  for s in range(0, len(steps), int(1 / DT)):
    chunk = steps[s:s + int(1 / DT)]
    results = [r for step_results, _ in chunk for r in step_results]
    sums = [sum(r[0] for r in results if isinstance(r, tuple)), sum(r for r in results if not isinstance(r, tuple)),
            *(int(sum(v)) for v in zip(*(state for _, state in chunk), strict=True))]
    seconds.append((hashlib.sha256(repr(chunk).encode()).hexdigest(), sums))
  return route, {"model": model, "seconds": seconds}


def pool():
  return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("spawn"))


def compare(base: dict, new: dict) -> int:
  fields = ["valid RX", "allowed TX", *(g.removeprefix("get_") for g in STATE)]
  diff = []
  for route in sorted(base):
    a, b = base[route], new[route]
    changed = [s for s, (x, y) in enumerate(zip(a["seconds"], b["seconds"], strict=True)) if x[0] != y[0]]
    if changed:
      what = sorted({fields[k] for s in changed for k, (x, y) in enumerate(zip(a["seconds"][s][1], b["seconds"][s][1], strict=True)) if x != y})
      diff.append(route)
      print(f"{route} {a['model']}: differs from {changed[0]} s on, in {len(changed)} of {len(a['seconds'])} s: {', '.join(what) or 'order only'}")
  print(f"{len(base)} routes, {sum(len(r['seconds']) for r in base.values())} s: " + (f"{len(diff)} routes differ" if diff else "no difference"))
  return int(bool(diff))


def main() -> int:
  # runs with the base's opendbc, so it only uses libsafety
  if sys.argv[1:2] == ["digest"]:
    with pool() as p:
      digests = dict(p.map(digest_route, sorted(Path(sys.argv[2]).glob("*.pkl"))))
    Path(sys.argv[3]).write_text(json.dumps(digests))
    return 0

  def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

  base_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD^"
  base, new = git("rev-parse", f"{base_ref}:opendbc_repo"), git("rev-parse", "HEAD:opendbc_repo")
  if base == new:
    print(f"opendbc is unchanged ({new}), nothing to compare")
    return 0
  print(f"comparing opendbc {base} -> {new}", flush=True)

  from opendbc.car.tests.routes import routes
  with tempfile.TemporaryDirectory() as tmp:
    w, odbc = Path(tmp), ROOT / "opendbc_repo"
    git("fetch", "-q", "--depth=1", "origin", base, cwd=odbc)
    git("worktree", "add", "-q", "--detach", str(w / "base"), base, cwd=odbc)
    try:
      # the base's safety code, built like this opendbc builds libsafety
      shutil.copy(odbc / "opendbc/safety/tests/libsafety/libsafety_py.py", w / "base/opendbc/safety/tests/libsafety/")
      (w / "routes").mkdir()
      with pool() as p:
        for e in filter(None, p.map(dump_route, [(i, r, w / "routes") for i, r in enumerate(routes)])):
          print(e)
      for name, path in (("base", w / "base"), ("new", odbc)):
        subprocess.check_call([sys.executable, __file__, "digest", w / "routes", w / f"{name}.json"], env={**os.environ, "PYTHONPATH": str(path)})
      return compare(json.loads((w / "base.json").read_text()), json.loads((w / "new.json").read_text()))
    finally:
      git("worktree", "remove", "--force", str(w / "base"), cwd=odbc)


if __name__ == "__main__":
  sys.exit(main())
