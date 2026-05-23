# ID4 MK1 fleet lateral-tracking analysis

Offline measurement pipeline that decides whether a thin per-car layer on top
of the existing VW MEB carcontroller can close the gap on poorly-tracking ID4
dongles, or whether no such layer is defensible. Produces per-dongle bucketed
residual statistics, lagged cross-correlations, alternative-hypothesis fits,
and a fleet-level decision with explicit recommendation.

## What it measures

For every gated livePose tick on every route, four desired/actual pairs are
accumulated into a `(7 speeds × 12 log-curvature buckets)` grid that matches
the sunnypilot dynamic_steering learner exactly:

| pair | desired | actual |
|---|---|---|
| **P1** | `carControl.actuators.curvature` (= controlsd's lag-adjusted target) | `yaw_rate / vEgo − roll_comp` |
| **P2** | `carControl.actuators.curvature` | `QFK_01.Curvature` (EPS rack measurement) |
| **P3** | `carOutput.actuatorsOutput.curvature` (after MEB additive correction) | yaw-rate-derived |
| **P4** | `carOutput.actuatorsOutput.curvature` | `QFK_01.Curvature` |

For each pair, per-bucket: count, signed/abs mean residual, sumsq, sign-stratified
left vs right means (the asymmetry the learner cannot see by design), and
saturation rate. Plus lagged Pearson correlation on a `[−500, +500] ms @ 50 ms`
grid (peak ≠ 0 ⇒ delay estimator is off, not a gain problem). Plus deadband
strata for `|desired| ∈ [0, 1e-5, 1e-4] rad/m`. Plus conditioning histograms
for `vEgo`, `lat_accel`, `liveParameters.steerRatio`, `liveParameters.stiffnessFactor`,
`liveDelay.lateralDelay`, EPS power, driver torque.

Gates mirror the sunnypilot learner exactly (lat-accel ≤ 1, |sin(roll)·g| ≤ 0.10,
yaw_rate_std < 1.0, vEgo ≥ MIN_SPEED, 2 s engagement / override buffer). Each
gate's rejection count is reported so per-route "gate yield" is observable.

## What it decides

Per dongle, seven hypotheses are fit against the pooled bucket sums:

- `H_null` — residual ≈ 0
- `H_gain` — `actual ≈ K · desired` with `0.2 < K < 1.5` (unphysical K → fit failure, not data)
- `H_lag` — xcorr peak at non-zero lag, `r ≥ 0.5`
- `H_asymmetry` — `K_pos − K_neg ≥ 0.05` (below threshold ⇒ collapse to H_gain)
- `H_deadband` — near-zero stratum mean residual ≥ 3× outer-stratum mean residual
- `H_speed`, `H_curvature` — placeholders, currently absorbed by H_null

Winner = lowest AIC, but **only** if its AIC is at least 2 below the next best.
Otherwise the dongle is labelled `mixed`. This is the alternative-hypothesis
check whose absence was identified as a prior failure mode.

Fleet-level conclusions are gated on `N ≥ 10 qualified dongles`. Below that,
`decision.md` carries a loud warning and the recommendation is `insufficient_data`.

## Usage

```sh
# Pipeline tests (no network, ~5s)
python3 -c "import sys; sys.path.insert(0,'.'); \
  from tools.lateral_maneuvers.id4_fleet.tests import test_features, test_hypotheses; \
  [getattr(m, n)() for m in (test_features, test_hypotheses) for n in dir(m) if n.startswith('test_')]"

# End-to-end smoke on the two reference routes (~30 s, downloads via LogReader)
python3 -m tools.lateral_maneuvers.id4_fleet.run \
    --manifest tools/lateral_maneuvers/id4_fleet/tests/reference_manifest.csv \
    --out /tmp/id4_fleet_smoke --workers 2

# Full fleet sweep (use most cores; leave a few for the system)
python3 -m tools.lateral_maneuvers.id4_fleet.run \
    --manifest path/to/your_manifest.csv \
    --out tools/lateral_maneuvers/id4_fleet/out \
    --workers 20 --timeout 300

# Re-aggregate without re-extracting (cheap, useful after iterating on hypotheses.py)
python3 -m tools.lateral_maneuvers.id4_fleet.run \
    --manifest /dev/null --out tools/lateral_maneuvers/id4_fleet/out --summarize-only
```

Manifest format:

```csv
dongle_id,route_id,branch
abcd1234,00000010--19b95d93b3,
abcd1234,0000000e--2d623b6df3,sunnypilot_pid
```

The `branch` column is free-text metadata, useful for splitting PID-on vs
PID-off routes when both are present on the same dongle.

## Outputs

```
out/
  per_route.parquet     # one row per (dongle, route): metadata + features blob (~50 KB each)
  per_dongle.parquet    # one row per dongle: pooled metadata + winning hypothesis + fits JSON
  failures.csv          # routes that didn't process and why (404, fingerprint, timeout, ...)
  decision.md           # top-level recommendation + leaderboard
  report.html           # matplotlib HTML with cross-dongle distribution + per-dongle heatmaps
```

`per_route.parquet` is **append-only and resumable**: re-running the orchestrator
will skip routes already present. The pipeline tolerates inaccessible routes
(deleted, 404, qlog-only, fingerprint mismatch, corrupt) without misleading the
aggregate; they get a status code and end up in `failures.csv`.

## Architecture

| file | purpose |
|---|---|
| `signals.py` | LogReader + `CANParser('vw_meb')` feed; yields one `Sample` per livePose tick |
| `gates.py` | Mirrors the sunnypilot learner's gates; tracks per-gate rejection counts |
| `features.py` | Streaming bucket accumulators + lagged xcorr; no full timelines in RAM |
| `extract.py` | Per-route worker; tolerates schema differences (openpilot5 vs sunnypilot) |
| `hypotheses.py` | AIC/BIC fits with explicit fit-failure labels for unphysical params |
| `aggregate.py` | Per-dongle pooling + bootstrap-free fleet decision logic |
| `offline_learner.py` | Replays a route through sunnypilot's `CurvatureEstimator` (~/openpilot10) |
| `report.py` | matplotlib + base64 HTML (style follows `generate_report.py`) |
| `run.py` | CLI orchestrator with `ProcessPoolExecutor`, per-route 5 min timeout, resumable parquet |
| `manifest.py` | CSV manifest loader + "already processed" check |

## Prior failure modes this design defends against

| Prior failure | Defense |
|---|---|
| N=2 routes / 1 dongle called a "fleet" | `fleet_conclusion_valid = (N ≥ 10)` with loud warning otherwise |
| `K > 1` results treated as data | `fit_gain` rejects `K ∉ (0.2, 1.5)` as fit failure |
| First-order fits where `a ∉ (0,1)` silently null `τ` | `fit_lag` rejects `r < 0.5` as fit failure |
| Per-route fits, no pooling | Pooling is explicit at per-dongle and fleet level; per-route only as input |
| Hand-decoded CAN bytes | `CANParser('vw_meb', ...)` with DBC signal names |
| No alternative hypothesis check | Five hypotheses scored with AIC/BIC; `mixed` label when none wins |
| Single-threaded, holds full timelines | Streaming accumulators (`< 5 s` ring buffers) + `ProcessPoolExecutor` |
| Silent saturation detector built on N=1 | Saturation reported as a feature, not as a flag |
| New state-machine classes around safety | None added; only data + a decision |
| Asymmetry overfitting from extra params | `H_asymmetry` requires `Δ K ≥ 0.05` to win, otherwise marked fit-failure |

## What this tool deliberately does **not** do

- Modify `opendbc_repo/opendbc/car/volkswagen/carcontroller.py`. A controller-side
  patch is downstream of the fleet result, not part of this measurement.
- Introduce `liveCurvatureParameters` to upstream cereal. The offline learner
  replay imports the sunnypilot estimator from `~/openpilot10` in-process.
- Run the full ~10 k-route corpus automatically. The orchestrator is the handoff;
  the user provides the manifest and chooses when to start the sweep.
- Make a decision when N < 10 qualified dongles. The recommendation in that case
  is always `insufficient_data` with a warning.

## Pipeline validation (current state)

- 13/13 unit tests pass (`test_features.py` + `test_hypotheses.py`)
- End-to-end orchestrator runs on the 2-route reference manifest, produces all
  artifacts (`per_route.parquet`, `per_dongle.parquet`, `failures.csv`,
  `decision.md`, `report.html`)
- The sunnypilot PID reference route extracts cleanly: 1402 gated samples,
  P1 RMS residual `1.7e-4 rad/m` in the 80-120 kph window, xcorr peak at
  `+200 ms` vs `liveDelay.lateralDelay = 234 ms` (residual phase + small gain
  mismatch consistent with the docstring intuition)
- Schema differences between openpilot5 and sunnypilot (`modelDesiredCurvature`,
  `rollCompensation`, `steeringSlightlyPressed`) are handled via safe getattr;
  schema-only sunnypilot fields are absent on openpilot5 routes without crashing
- The stock-Travel-Assist reference route returns `status="no_engagement"` (correct
  behavior: openpilot is not active in TA mode, so there are no engaged samples
  to gate through). To validate the "PID < stock" comparison, add a stock-openpilot
  route (not stock TA) to the manifest.

## Next steps not in this session

- Build the actual ~10 k-route manifest (comma connect API or a saved file) and run the sweep.
- Add `offline_learner.replay()` results to the report so the converged learner
  bias and the offline batch fit are visible side by side per dongle.
- Bootstrap CIs on per-dongle scalar tracking score for proper inference.
- If the fleet result picks `per_dongle_gain_scalar`, propose a concrete patch
  to `carcontroller.py` that reads a small per-VIN gain table; if it picks
  `fix_delay_estimator`, propose investigation steps in `selfdrive/locationd/`;
  if `negative_result_asymmetry`, write up the negative result as the deliverable.
