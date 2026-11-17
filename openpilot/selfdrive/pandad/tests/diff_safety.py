#!/usr/bin/env python3
# replays the car test routes through BASE's and this opendbc's safety, reports every changed RX and TX result and safety state value
#   diff_safety.py [BASE]    BASE defaults to HEAD^, which is master on a PR's merge commit
import multiprocessing
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "diff_safety.txt"
DT = 0.01
BUSES = 3
# a TX message's bus, like cereal's can src
TX = 128
STATE = ("get_controls_allowed", "get_relay_malfunction", "get_gas_pressed_prev", "get_brake_pressed_prev", "get_regen_braking_prev",
         "get_steering_disengage_prev", "get_cruise_engaged_prev", "get_vehicle_moving", "get_acc_main_on", "get_torque_meas_min",
         "get_torque_meas_max", "get_torque_driver_min", "get_torque_driver_max", "get_angle_meas_min", "get_angle_meas_max",
         "get_curvature_meas_min", "get_curvature_meas_max", "get_vehicle_speed_min", "get_vehicle_speed_max", "get_desired_curvature_last",
         "get_honda_fwd_brake")
BOOL_STATE = {"controls_allowed", "relay_malfunction", "gas_pressed_prev", "brake_pressed_prev", "regen_braking_prev", "steering_disengage_prev",
              "cruise_engaged_prev", "vehicle_moving", "acc_main_on", "honda_fwd_brake"}


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
      # sent and blocked TX, so a check that allows more shows up too
      elif tx and m.src >= TX and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), TX + m.src % 4))
  return cfg.safetyModel.raw, cfg.safetyParam, frames


def replay(mode: int, param: int, frames, on_tick=None):
  # the result of each message and the safety state after each 10 ms step
  from opendbc.safety.tests.libsafety import libsafety_py
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(mode, param) == 0
  safety.set_alternative_experience(0)
  # older opendbc names it safety_tick_current_safety_config
  tick = safety.safety_tick if hasattr(safety, "safety_tick") else safety.safety_tick_current_safety_config

  steps = []
  for i, frame in enumerate(frames):
    safety.set_timer(int(i * DT * 1e6))
    if i % int(1 / DT) == 0:
      tick()
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
      pickle.dump((route.route, str(route.car_model), *load(route, tx=True)), f)
  except Exception as e:
    return f"ERROR {route.car_model} - {route.route}: can't load, {e!r}"
  return None


def results_route(args):
  # every message's result (RX valid or TX allowed), where each RX is forwarded, and the state after each step
  path, out = args
  with open(path, "rb") as f:
    route, model, mode, param, frames = pickle.load(f)
  from opendbc.safety.tests.libsafety import libsafety_py
  if libsafety_py.libsafety.set_safety_hooks(mode, param) != 0:
    np.savez(out / path.with_suffix(".npz").name, mode=mode)
    return
  steps = replay(mode, param, frames)
  results = [r for step_results, _ in steps for r in step_results]
  np.savez(out / path.with_suffix(".npz").name,
           ok=np.array([r[0] if isinstance(r, tuple) else r for r in results], dtype=bool),
           fwd=np.array([r[1] if isinstance(r, tuple) else -2 for r in results], dtype=np.int16),
           state=np.array([state for _, state in steps], dtype=np.float64))


def pool():
  return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("spawn"))


def value(v: float, name: str):
  if name in BOOL_STATE:
    return bool(v)
  return int(v) if v == int(v) else round(v, 3)


def message_diffs(frames, a, b) -> list[str]:
  # RX valid, RX forward and TX allowed, grouped by message and change
  events = [(i, addr, bus) for i, frame in enumerate(frames) for addr, _, bus in frame]
  changes = Counter()
  first = {}
  for k in np.flatnonzero((a["ok"] != b["ok"]) | (a["fwd"] != b["fwd"])):
    i, addr, bus = events[k]
    if bus >= TX:
      key = (f"TX 0x{addr:x} bus {bus - TX} allowed", bool(a["ok"][k]), bool(b["ok"][k]))
    elif a["ok"][k] != b["ok"][k]:
      key = (f"RX 0x{addr:x} bus {bus} valid", bool(a["ok"][k]), bool(b["ok"][k]))
    else:
      key = (f"RX 0x{addr:x} bus {bus} forward", int(a["fwd"][k]), int(b["fwd"][k]))
    changes[key] += 1
    first.setdefault(key, i)
  return [f"\n  {what}: {old} -> {new} ({n} msgs, from frame {first[(what, old, new)]})" for (what, old, new), n in sorted(changes.items())]


class StateWrap:
  """Adapter so car_diff reads one safety state value"""
  def __init__(self, name: str, v):
    self._d = {name: v}
  def to_dict(self) -> dict:
    return self._d


def state_diffs(a, b) -> list[str]:
  from opendbc.car.tests.car_diff import format_diff
  lines = []
  for k, getter in enumerate(STATE):
    name = getter.removeprefix("get_")
    old, new = a["state"][:, k], b["state"][:, k]
    frames = np.flatnonzero(old != new)
    if not len(frames):
      continue
    diffs = [(name, int(i), (value(old[i], name), value(new[i], name)), int(i * DT * 1e9)) for i in frames]
    ref = [(int(i * DT * 1e9), StateWrap(name, value(v, name))) for i, v in enumerate(old)]
    states = [StateWrap(name, value(v, name)) for v in new]
    lines.append(f"\n  {name} ({len(diffs)} diffs)")
    lines += format_diff(diffs, ref, states, name)
  return lines


def diff_report(dumps: Path, base: Path, new: Path, errors: list[str]) -> int:
  changed, new_cars, n_passed = [], [], 0
  for dump in sorted(dumps.glob("*.pkl"), key=lambda p: int(p.stem)):
    with open(dump, "rb") as f:
      route, model, mode, _, frames = pickle.load(f)
    a, b = np.load(base / f"{dump.stem}.npz"), np.load(new / f"{dump.stem}.npz")
    if "ok" not in b:
      errors.append(f"ERROR {model} - {route}: this opendbc has no safety mode {mode}")
    elif "ok" not in a:
      new_cars.append(f"\nNEW {model} - {route}: master's opendbc has no safety mode {mode}")
    else:
      lines = message_diffs(frames, a, b) + state_diffs(a, b)
      if lines:
        changed.append([f"\n{model} - {route}"] + lines)
      else:
        n_passed += 1

  icon = "⚠️" if changed or errors else "✅"
  lines = [
    "## Safety replay diff report",
    "Replays every car test route through master's and this PR's panda safety and compares each RX and TX result and the safety state.",
    "Please review any changes carefully to ensure they are expected.\n",
    f"{icon}  {len(changed)} changed, {n_passed} passed, {len(errors)} errors" + (f", {len(new_cars)} new" if new_cars else ""),
  ]
  lines += [f"\n{e}" for e in errors] + new_cars
  if changed:
    lines.append("<details><summary><b>Show changes</b></summary>\n\n```")
    for route_lines in changed:
      lines += route_lines
    lines.append("```\n</details>")
  REPORT.write_text("\n".join(lines) + "\n")
  print("\n".join(lines))
  return int(bool(changed or errors))


def main() -> int:
  # runs with the base's opendbc, so it only uses libsafety
  if sys.argv[1:2] == ["results"]:
    dumps, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir()
    with pool() as p:
      list(p.map(results_route, [(d, out) for d in sorted(dumps.glob("*.pkl"))]))
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
      (w / "routes").mkdir()
      with pool() as p:
        errors = [e for e in p.map(dump_route, [(i, r, w / "routes") for i, r in enumerate(routes)]) if e]
      for name, path in (("base", w / "base"), ("new", odbc)):
        subprocess.check_call([sys.executable, __file__, "results", w / "routes", w / f"{name}_results"], env={**os.environ, "PYTHONPATH": str(path)})
      return diff_report(w / "routes", w / "base_results", w / "new_results", errors)
    finally:
      git("worktree", "remove", "--force", str(w / "base"), cwd=odbc)


if __name__ == "__main__":
  sys.exit(main())
