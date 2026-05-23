# Fleet residual analysis - VW MEB ID4_MK1

Per-car lateral tracking quality varies on VW MEB ID4_MK1 running openpilot.
A sunnypilot fork patches it with a PID; comma does not want a PID because the
EPS plant is not understood. The open question is whether the variation is a
systematic plant property (gain, lag, deadband, asymmetry), a per-car
calibration issue (`liveParameters.steerRatio`, stiffness, alignment), or
unstructured noise.

This tool produces evidence to answer that question across the fleet without
committing to a remedy in advance.

## What the prior art does

The sunnypilot dynamic_steering learner
(`selfdrive/locationd/curvatured.py` in `~/openpilot10`) is a per-(speed,
|curvature|) bias estimator. It compares
`controlsState.modelDesiredCurvature` to a yaw-rate-derived "actual"
curvature, gated and lag-aligned, and folds the error into an EMA bias per
bucket. It does **not** see the EPS-measured rack curvature (`QFK_01`), so it
cannot distinguish "the EPS executes our command but the vehicle responds
differently" from "the EPS doesn't execute what we asked." It also implicitly
assumes the residual is a steady-state gain phenomenon - lag, deadband,
asymmetry are out of scope.

## What this tool measures

For each route, five residual surfaces on the same `(7 speed) x (12 |curv|) x
(2 sign)` bucket grid the learner uses:

| Residual | Subtracts | Meaning |
|---|---|---|
| `R_model_yaw`   | `model_raw - yaw_actual` | learner view, outer envelope |
| `R_model_qfk`   | `model_raw - QFK_01`     | model vs rack-measured |
| `R_hca_qfk`     | `HCA_03   - QFK_01`     | EPS execution (does the rack do what we asked?) |
| `R_hca_yaw`     | `HCA_03   - yaw_actual` | EPS-to-plant (did the rack-as-commanded produce the expected yaw?) |
| `R_smooth_loss` | `model_raw - actuators.curvature` | controller-internal smoothing/lag-adjustment |

Plus per-speed-band lag estimates (cross-correlation residual after liveDelay
pre-shift), a deadband signature in the smallest |curv| buckets, and a
left/right asymmetry signature.

Why this decomposition: if `R_hca_qfk` is small and `R_hca_yaw` is large, the
EPS is faithful and the residual is between the rack and the plant
(calibration/stiffness). If `R_hca_qfk` is large, the EPS has its own
dynamics the controller doesn't model. The sunnypilot learner pools both
into `R_model_yaw` and cannot separate them.

Sign convention matches `carstate.py` so HCA and QFK are directly
comparable; both fields use `-Curvature * (1, -1)[VZ]` decoding.

## Layout

    extract.py        per-route worker (LogReader -> pickle summary)
    run_fleet.py      ProcessPoolExecutor over a route list (CSV/text)
    analyze.py        load pickles -> text + HTML report
    out/              pickles, failed.log, progress.jsonl, report.html
    routes.example.csv   example route list

The pipeline never holds full 100Hz timelines for a route in memory; each
worker emits a ~30 KB pickle of sufficient statistics and discards the rest.
For 10 K routes that is ~300 MB of pickles.

## Usage

### One route, fast iteration

    python -m tools.lateral_maneuvers.fleet_residuals.extract \
        f73c01590368ee5b/0000000e--2d623b6df3/a

### Fleet

Provide a route list (one `dongle/route[/segment]` per line; default segment
is `a` = rlogs with qlog fallback):

    python -m tools.lateral_maneuvers.fleet_residuals.run_fleet \
        --routes routes.csv --workers 16

Resumable: routes whose pickle already exists are skipped. Crashes are
categorised and logged to `out/failed.log` with traceback.

### Analyse

    python -m tools.lateral_maneuvers.fleet_residuals.analyze \
        'tools/lateral_maneuvers/fleet_residuals/out/*.pkl'

Single pickle -> per-route text + heatmap HTML. Multiple pickles -> fleet
summary table and per-residual median/p10/p90 across routes. The analyser
refuses to compute population statistics with fewer than 10 unique dongles
and prints an "INSUFFICIENT POPULATION" message instead.

## Decision tree

The deliverable is whichever of these the fleet data supports - it is not
pre-decided:

| Finding | Suggested remedy |
|---|---|
| Residual concentrates in `R_model_yaw / R_hca_yaw` only, correlates with `liveParameters.steerRatio` drift | Recommend forcing a steerRatio recalibration on outliers; **no controller change** |
| Residual is a stable per-fingerprint scalar across most dongles, with low intra-dongle variance | Propose a `lateralTuning` table entry; not a daemon |
| Residual structure matches the learner's per-bucket assumption and the learner converges correctly | Propose a stripped-down upstream port with the plant evidence comma asked for |
| Residual is a phase lag at given speed (lag table differs from `liveDelay`) | Tooling for a `liveDelay` study; not a curvature patch |
| Residual is a deadband near zero or a left/right asymmetry | Different remedy; not a per-bucket gain |
| Residual is unstructured noise across the fleet | Negative result; document it and stop |

## Known limitations

- Stock-TA routes show zero engaged samples (correct, openpilot was not
  driving) - the tool reports this cleanly. To measure stock-TA's tracking,
  HCA_03 would need to be parsed from `can` rather than `sendcan`; add a
  flag if this becomes useful.
- `carControl.rollCompensation` is deprecated. Roll compensation is
  reconstructed as `g * liveParameters.roll / vEgo^2` (dominant term;
  slip-factor correction is small at MEB highway speeds).
- Yaw rate is taken from `livePose.angularVelocityDevice.z` without the
  full `PoseCalibrator` transform. For a calibrated dongle the difference
  is small; if a route's residuals look pathological compared to siblings,
  reconsider.
- LogReader messages may not be globally time-sorted across segments. The
  extractor sorts each signal before resampling; this is essential.
- The bucket grid (`SPEED_ANCHORS`, `CURV_EDGES`) matches the learner so
  the offline batch fit and the learner's published `liveCurvatureParameters`
  surface are directly comparable. If the residual concentrates outside
  this grid on the fleet, the grid itself is part of the finding.

## What the user still needs to supply

A fleet route list. The two reference routes in `routes.example.csv` are
useful for pipeline calibration but neither is a "comma openpilot tracking
poorly" example - one is stock TA (no openpilot at all) and one is
sunnypilot PID (not comma controller). Seeding the fleet run with a handful
of known-bad-tracking comma openpilot dongles would let the pipeline
sanity-check that it picks up the symptom before scaling out.
