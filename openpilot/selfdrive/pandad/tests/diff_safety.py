#!/usr/bin/env python3
# replays the car test routes through BASE's and this opendbc's safety, reports every changed RX and TX result and safety state value
#   diff_safety.py [BASE]          BASE defaults to HEAD^, which is master on a PR's merge commit
#   diff_safety.py check [BASE]    prints changed=true when opendbc differs from BASE's
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from opendbc.car import DT_CTRL as DT

# GitHub comments hold 65536 characters, and this shares one with the process replay report
MAX_REPORT = 50000
# the panda's CAN buses, a log's can src is bus + 4 * panda
BUSES = 3
# the panda marks sent CAN with src + 128 and blocked with src + 192
TX = 128


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
      if m.src < TX and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), m.src % 4))
      # sent and blocked TX, so a check that allows more shows up too
      elif tx and m.src >= TX and m.src % 4 < BUSES:
        frames[i].append((m.address, bytes(m.dat), TX + m.src % 4))
  return cfg.safetyModel.raw, cfg.safetyParam, frames


def replay(mode: int, param: int, frames, on_tick=None):
  # the result of each message and the whole safety state (safety_a) after each 10 ms step
  from opendbc.safety.tests.libsafety import libsafety_py
  safety = libsafety_py.libsafety
  assert safety.set_safety_hooks(mode, param) == 0
  safety.set_alternative_experience(0)
  # older opendbc names it safety_tick_current_safety_config
  tick = safety.safety_tick if hasattr(safety, "safety_tick") else safety.safety_tick_current_safety_config
  state = libsafety_py.ffi.buffer(libsafety_py.ffi.addressof(safety, "safety_a"))

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
    steps.append((results, state[:]))
  return steps


def dump_route(args):
  i, route, out = args
  try:
    with open(out / f"{i}.pkl", "wb") as f:
      pickle.dump((route.route, str(route.car_model), *load(route, tx=True)), f)
  except Exception as e:
    return f"{route.car_model} - {route.route}: can't load, {e!r}"
  return None


def state_layout(ffi):
  # every field of the safety state with its offset and type, pointers left out since they differ between processes
  fields = []

  def walk(t, path, offset):
    if t.kind == "struct":
      for name, f in t.fields:
        walk(f.type, f"{path}.{name}" if path else name, offset + f.offset)
    elif t.kind == "array":
      for i in range(t.length):
        walk(t.item, f"{path}[{i}]", offset + i * ffi.sizeof(t.item))
    elif t.kind in ("primitive", "enum"):
      size = ffi.sizeof(t)
      if t.cname == "_Bool":
        dtype = "?"
      else:
        # cffi only converts float types with float()
        try:
          float(ffi.cast(t, 0))
          dtype = f"f{size}"
        except TypeError:
          dtype = f"i{size}" if int(ffi.cast(t, -1)) < 0 else f"u{size}"
      fields.append((path.removeprefix("state."), offset, size, dtype))

  walk(ffi.typeof("SafetyCopy"), "", 0)
  return fields


def results_route(args):
  # every message's result (RX valid or TX allowed), where each RX is forwarded, and each safety state field after each step
  path, out, build = args
  with open(path, "rb") as f:
    route, model, mode, param, frames = pickle.load(f)
  from opendbc.safety.tests.libsafety import libsafety_py
  # a copy of the build is a fresh library, no safety state carries over from another route
  fd, copy = tempfile.mkstemp(suffix=".so")
  os.close(fd)
  shutil.copy(build, copy)
  libsafety_py.load(copy)
  os.unlink(copy)
  if libsafety_py.libsafety.set_safety_hooks(mode, param) != 0:
    np.savez(out / path.with_suffix(".npz").name, mode=mode)
    return
  steps = replay(mode, param, frames)
  results = [r for step_results, _ in steps for r in step_results]
  raw = np.frombuffer(b"".join(state for _, state in steps), dtype=np.uint8).reshape(len(steps), -1)
  arrays: dict[str, Any] = {f"state:{name}": raw[:, offset:offset + size].copy().view(dtype)[:, 0]
                            for name, offset, size, dtype in state_layout(libsafety_py.ffi)}
  arrays["ok"] = np.array([r[0] if isinstance(r, tuple) else r for r in results], dtype=bool)
  arrays["fwd"] = np.array([r[1] if isinstance(r, tuple) else -2 for r in results], dtype=np.int16)
  np.savez_compressed(out / path.with_suffix(".npz").name, **arrays)


def pool():
  return ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("spawn"))


def value(v):
  return bool(v) if isinstance(v, np.bool_) else v.item()


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
  """Adapter so car_diff reads one safety state field, dotted names are nested like in a struct"""
  def __init__(self, name: str, v):
    *parents, leaf = name.split(".")
    self._d = d = {}
    for p in parents:
      d = d.setdefault(p, {})
    d[leaf] = v
  def to_dict(self) -> dict:
    return self._d


def state_diffs(a, b, full: bool = False) -> list[str]:
  # full: every changed frame instead of the first ones, for the report in the artifact
  from opendbc.car.tests.car_diff import format_diff, format_numeric_diffs
  lines = []
  for key in sorted(a.keys() | b.keys()):
    if not key.startswith("state:"):
      continue
    name = key.removeprefix("state:")
    if key not in a or key not in b:
      lines.append(f"\n  {name}: only in {'master' if key in a else 'this PR'}")
      continue
    old, new = a[key], b[key]
    frames = np.flatnonzero(old != new)
    if not len(frames):
      continue
    diffs = [(name, int(i), (value(old[i]), value(new[i])), int(i * DT * 1e9)) for i in frames]
    ref = [(int(i * DT * 1e9), StateWrap(name, value(v))) for i, v in enumerate(old)]
    states = [StateWrap(name, value(v)) for v in new]
    lines.append(f"\n  {name} ({len(diffs)} diffs)")
    if full and not isinstance(diffs[0][2][0], bool):
      for i in range(0, len(diffs), 10):
        lines += format_numeric_diffs(diffs[i:i + 10])
    else:
      lines += format_diff(diffs, ref, states, name)
  return lines


def diff_report(dumps: Path, base: Path, new: Path, errors: list[str]) -> int:
  changed, changed_full, new_cars, n_passed = [], [], [], 0
  for dump in sorted(dumps.glob("*.pkl"), key=lambda p: int(p.stem)):
    with open(dump, "rb") as f:
      route, model, mode, _, frames = pickle.load(f)
    a, b = dict(np.load(base / f"{dump.stem}.npz")), dict(np.load(new / f"{dump.stem}.npz"))
    if "ok" not in b:
      errors.append(f"{model} - {route}: this opendbc has no safety mode {mode}")
    elif "ok" not in a:
      new_cars.append(f"\nNEW {model} - {route}: master's opendbc has no safety mode {mode}")
    else:
      messages, states = message_diffs(frames, a, b), state_diffs(a, b)
      if messages or states:
        changed.append([f"\n{model} - {route}"] + messages + states)
        changed_full.append([f"\n{model} - {route}"] + messages + state_diffs(a, b, full=True))
      else:
        n_passed += 1

  from openpilot.common.basedir import BASEDIR
  from openpilot.selfdrive.test.diff_format import format_report

  def fmt(shown, notes):
    return format_report("Safety replay diff report", "Replays every car test route through this PR's panda safety and compares the behavior to master.",
                         shown, n_passed, errors, new_cars + notes, n_changed=len(changed))

  (Path(BASEDIR) / "diff_safety_full.txt").write_text(fmt(changed_full, []))

  # the comment shows as many routes as fit, the full report is in the job's artifact
  notes = []
  if changed and "GITHUB_RUN_ID" in os.environ:
    run = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    notes.append(f"\nEvery changed frame: diff_safety_full.txt in {run}")
  n = len(changed)
  while n and len(fmt(changed[:n] + [[f"\n(... {len(changed) - n} more routes)"]], notes)) > MAX_REPORT:
    n -= 1
  shown = changed if n == len(changed) else changed[:n] + [[f"\n(... {len(changed) - n} more routes)"]]
  report = fmt(shown, notes)
  print(report)
  (Path(BASEDIR) / "diff_safety.txt").write_text(report)
  return int(bool(changed or errors))


def main() -> int:
  # runs with the base's opendbc, so it only uses libsafety
  if sys.argv[1:2] == ["results"]:
    dumps, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir()
    from opendbc.safety.tests.libsafety import libsafety_py
    build = libsafety_py._build_libsafety()
    with pool() as p:
      list(p.map(results_route, [(d, out, build) for d in sorted(dumps.glob("*.pkl"))]))
    return 0

  from openpilot.common.basedir import BASEDIR
  root = Path(BASEDIR)

  def git(*args, cwd=root):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

  check = sys.argv[1:2] == ["check"]
  base_ref = (sys.argv[2:] if check else sys.argv[1:]) or ["HEAD^"]
  base, new = git("rev-parse", f"{base_ref[0]}:opendbc_repo"), git("rev-parse", "HEAD:opendbc_repo")
  # for a GitHub step's outputs
  if check:
    print(f"changed={str(base != new).lower()}")
    return 0
  if base == new:
    print(f"opendbc is unchanged ({new}), nothing to compare")
    return 0
  print(f"comparing opendbc {base} -> {new}", flush=True)

  from opendbc.car.tests.routes import routes
  with tempfile.TemporaryDirectory() as tmp:
    w, odbc = Path(tmp), root / "opendbc_repo"
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
