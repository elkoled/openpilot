# `vw_id4_lateral` — Fleet investigation of ID4_MK1 lateral tracking quality

Offline analysis pipeline that characterises per-car variation in VW MEB
(VOLKSWAGEN_ID4_MK1) lateral tracking on openpilot, and tests whether the
sunnypilot `dynamic_steering` learner is the right answer for any cars in
the bad-tracking tail.

This package writes data and a report. It does **not** ship a controller
patch — the decision tree in the report says whether one is defensible
and, if so, what it should look like.

## Layout

```
tools/vw_id4_lateral/
  manifest.py          # Stage 1 — robust qlog-probe of every route
  grid.py              # Bucket aggregator + grid definitions
  learner_replay.py    # Stage 3 — vendored CurvatureD learner + offline batch fit
  extract_segment.py   # Stage 2 — per-segment 100Hz align + bucket aggregate
  run_fleet.py         # Multiprocessing fleet driver (resumable)
  analyze.py           # Stage 4 — cross-car stats & hypothesis battery
  report.py            # Stage 5 — single HTML report
  test_extract.py      # Layered tests (synthetic + reference-route + render)
```

## Run

```bash
# 1) Manifest — probe each route once for fingerprint, VIN, engaged time
python -m tools.vw_id4_lateral.manifest routes.txt --out manifest.jsonl --workers 8

# 2) Fleet feature extraction — RLOG-level, per-segment, parallel, resumable
python -m tools.vw_id4_lateral.run_fleet --manifest manifest.jsonl \
       --out per_segment.jsonl --workers 8

# 3) Cross-car aggregation + hypothesis battery
python -m tools.vw_id4_lateral.analyze --in per_segment.jsonl --out fleet_stats.pkl

# 4) HTML report
python -m tools.vw_id4_lateral.report --stats fleet_stats.pkl --out fleet_report.html
```

`routes.txt`: one `dongle_id/route_id` per line (or `dongle|route`). Blank
lines and `#` comments are ignored.

## Output formats

- `manifest.jsonl` — one JSON row per probed route. `accessible=true` means
  ID4_MK1 fingerprint + ≥ ~60 s engaged in qlog. The `has_live_curvature_parameters`
  flag separates dongles running the sunnypilot dynamic_steering learner
  from comma-stock dongles.
- `per_segment.jsonl` — one JSON row per accessible route with bucket
  aggregates, per-segment scalars, learner-replay result, and offline
  batch-fit result. ~5 KB/route.
- `fleet_stats.pkl` — cross-car aggregates and hypothesis-battery numbers.
  Lives ≪ 50 MB even for the full ~10k fleet.
- `fleet_report.html` — final report. Self-contained (matplotlib base64
  embedded as in `tools/longitudinal_maneuvers/generate_report.py`).

## Resource budget

- Per-segment memory: a single segment's 100 Hz arrays only (~400 KB peak),
  then dropped before return. No cross-segment timeline accumulation.
- Fleet runtime on an 8-core workstation: ~2 hours for ~10k segments.
  Resumable — re-running skips routes already in `per_segment.jsonl`.
- Cache: routes are fetched via `tools.lib.logreader.LogReader` which uses
  `~/.commacache/local/` automatically.

## Method (one paragraph)

The MEB controller closes the **rack loop**:
`apply_curvature = actuators.curvature + (CS.curvature_meas - CC.currentCurvature)`
(`opendbc_repo/opendbc/car/volkswagen/carcontroller.py:82`). The
dynamic_steering learner closes the **vehicle-response loop** instead:
`actual = yaw_rate/vEgo - sin(roll)·g/vEgo`. The two ground truths can
disagree; this pipeline computes both and uses the vehicle-response loop
for residual statistics, matching the learner's definition so the
comparison is apples-to-apples. Residuals are aggregated on two grids: the
learner's 7×12 supported region (the only region where it can apply a
correction) and an extended 7×16 grid that goes ~16× higher in curvature
and ~14% higher in speed. The fraction of |residual mass| inside vs
outside the supported region is one of the discriminators in the decision
tree — if most residual lives outside, the learner cannot help by
construction.

## Safeguards

The pipeline was designed against a list of prior failure modes the user
laid out explicitly. Concretely:

| Prior failure mode | How this pipeline avoids it |
|---|---|
| N=2 routes from 1 dongle | All figures annotated with N_cars; plots hidden when N < 5 |
| `K ∉ (0, 1)` overfits reported as data | Per-speed gain fits enforce `gain ∈ (0.3, 1.7)`; out-of-bounds excluded, not silently reported |
| Per-route iteration, no fleet pooling | Bucket aggregates are explicitly pooled in `analyze.py`; `extract_segment.py` never reports population stats |
| Hand-decoded CAN bytes | All CAN goes through `CANParser('vw_meb', ...)` |
| No alternative-hypothesis check | Hypothesis battery in `analyze.py` runs pre-registered tests: gain, lag, deadband, left/right asymmetry, transient |
| 100 Hz timeline memory bombs | Per-segment output is the (S, C) bucket aggregate, not the timeline; multiprocessing workers hold one segment at a time |
| New state-machine classes in the controller | None introduced — vendored learner runs offline only |
| Silent saturation detectors | The pipeline ships *no* detectors and *no* controller patch — only data |

## Vendored code

`learner_replay.py` vendors the bucket math and per-bucket EMA update from
`openpilot10/selfdrive/locationd/curvatured.py` (branch
`virtual/dynamic_steering`, HEAD sha `febd9128d8e03b551b47d1a69b778aae6edf1136`,
file last-touch `ba93c4129cd5bfb1a9b28d59a7b7879d6fec7c3a`). The vendored
math is pure numpy — no cereal, no Params, no PoseCalibrator dependency —
so it replays offline without the daemon's messaging plumbing. Provenance
is in the file header. Pre-registered unit tests in `test_extract.py`
check the vendored math against the production daemon's specified
behaviour (direction-projected error, ±50% relative cap, MAX_SAMPLES
saturation, two-anchor speed interpolation).

## Tests

```bash
# All synthetic + render tests, no network needed:
python -m pytest tools/vw_id4_lateral/test_extract.py -v -m "not slow" --noconftest

# Network-dependent reference-route tests (the two anchor routes the user
# supplied — Stock TA and Sunnypilot PID); skipped if network unavailable:
python -m pytest tools/vw_id4_lateral/test_extract.py -v --noconftest
```

`--noconftest` is needed because the openpilot top-level conftest imports
`msgq.ipc_pyx`, which requires a scons build. This pipeline doesn't need
msgq at all.

## Reference routes

Same dongle, two configurations:

- `f73c01590368ee5b/00000010--19b95d93b3` — Stock VW Travel Assist
- `f73c01590368ee5b/0000000e--2d623b6df3` — sunnypilot PID engaged

Useful for sanity-checking that `extract()` produces non-degenerate output
on real data before running the full fleet.
