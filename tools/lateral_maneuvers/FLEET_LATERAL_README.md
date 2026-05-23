# Fleet lateral tracking analysis (VW MEB / ID4_MK1)

Tools for characterizing per-car variation in lateral tracking quality across a
fleet of openpilot dongles.  Reads engaged-route logs once, accumulates
sufficient statistics in `(speed × signed_curvature)` buckets, then aggregates
across cars to expose the cross-dongle distribution — not just the fleet mean.

## Why these files exist

The existing MEB controller (`opendbc/car/volkswagen/carcontroller.py:82`) does:

    apply_curvature = actuators.curvature + (CS.curvature_meas - CC.currentCurvature)

Tracking quality on `VOLKSWAGEN_ID4_MK1` varies car-to-car.  Sunnypilot's
`virtual/dynamic_steering` branch adds an active learner on top
(`selfdrive/locationd/curvatured.py`); mainline rejects that approach because
the EPS plant is not understood.  Whether the gap is a systematic plant
property or a per-car calibration issue is the open question.

These tools answer the question with data:

1. **`fleet_lateral_extract.py`** — per-route streaming summarizer.  Processes
   one rlog/qlog, gates samples (engagement, override, roll, yaw uncertainty,
   speed floor), aligns desired vs actual with `liveDelay.lateralDelay`, and
   emits sufficient statistics for the three truth sources.
2. **`fleet_lateral_run.py`** — multi-process driver.  Reads a route-list TSV,
   parallelizes extraction, persists per-route NPZ + manifest, tolerates all
   common failure modes (LogsUnavailable, capnp errors, wrong fingerprint,
   empty qlogs, timeouts).
3. **`fleet_lateral_aggregate.py`** — cross-dongle aggregator.  Pools the
   per-route NPZ by dongle, computes per-dongle pooled residuals and a
   cross-dongle distribution per `(speed, curvature)` bucket.

## Schema (per-route NPZ)

| Array              | Shape          | Meaning                                     |
|--------------------|----------------|---------------------------------------------|
| `n`                | (S, C)         | Accepted samples in bucket                  |
| `desired_sum`      | (S, C)         | Σ desired curvature                          |
| `desired_sumsq`    | (S, C)         | Σ desired²                                   |
| `vmcurv_sum`       | (S, C)         | Σ CC.currentCurvature (VM-derived)           |
| `vmcurv_sumsq`     | (S, C)         |                                              |
| `hca_sent_sum/sumsq/n` | (S, C)     | Σ commanded (carOutput.actuatorsOutput)      |
| `truth_n`          | (S, C, T)      | Per-truth valid sample count                 |
| `truth_sum`        | (S, C, T)      | Σ truth (per source)                         |
| `resid_td_sum`     | (S, C, T)      | Σ (truth − desired) — tracking error         |
| `resid_tv_sum`     | (S, C, T)      | Σ (truth − vmcurv) — actuation error         |
| `resid_ht_sum`     | (S, C, T)      | Σ (hca_sent − truth) — overcommand           |
| `header_json`      | string         | dongle, route, fingerprint, VIN, branch,     |
|                    |                | mean liveParams.steerRatio / stiffness,      |
|                    |                | mean liveDelay.lateralDelay, ...             |

Truth sources `T = ["pose_yaw", "cs_yaw", "qfk_rack"]`:
- `pose_yaw` — `CC.angularVelocity[2] / vEgo` (already calibrated by controlsd).
  Falls back to raw `livePose.angularVelocityDevice.z` if CC value is missing.
  **Canonical truth source on VW.**
- `cs_yaw` — `carState.yawRate / vEgo`.  Not populated on VW (always 0) on
  many branches; treated as missing when exactly 0.
- `qfk_rack` — `QFK_01.Curvature` from CAN (EPS rack feedback).  Decoded with
  the MEB sign convention.  Prior single-route analysis showed R² ≈ 0.1 vs
  pose-derived truth — noisy.

Bin edges:
- Speed: `[5, 10, 15, 20, 25, 30, 35, 1000] m/s` (7 bins; below 5 m/s skipped).
- Curvature: signed log-spaced, `[-1, ±8e-3, ±4e-3, ±2e-3, ±1e-3, ±5e-4,
  ±2.5e-4, ±1e-4, +1]` (15 bins).  Asymmetric L/R analyses just compare
  positive vs negative bins.

## Tomorrow's workflow

1. Drop the private route list at any path, e.g. `~/curv_analysis/id4_routes.txt`.
   Format (whitespace-separated; `#` comments allowed):

   ```
   <dongle_id_a>  <route_id_a>
   <dongle_id_b>  <route_id_b>
   …
   ```
   or `<dongle_id>/<route_id>` per line.

2. Launch:

   ```bash
   python tools/lateral_maneuvers/fleet_lateral_run.py \
     ~/curv_analysis/id4_routes.txt \
     /home/batman/curv_analysis/fleet_summaries \
     --workers 16 --timeout 300
   ```

   Per-route runtime budget: ~15–60 s on qlog, ~30–200 s on rlog with full CAN.
   With 16 workers, 10 k routes ≈ 3–6 hours wall.

3. Aggregate:

   ```bash
   python tools/lateral_maneuvers/fleet_lateral_aggregate.py \
     /home/batman/curv_analysis/fleet_summaries \
     --min_dongle_engaged_s 600
   ```

   Produces `per_dongle.tsv`, `cross_dongle.tsv`, `per_dongle_bucket.npz`,
   `fleet_summary.txt`.

## Smoke test (already run on 7 routes / 2 dongles)

`/tmp/fleet_smoke` from `tools/lateral_maneuvers/{...}` shows the toolchain
end-to-end.  Notable finding from the smoke alone: dongle
`aebd8f1d4ea16066` straddles **two VINs** and **four branches**
(`master`, `id`, `id42`, `id4-merged`).  The aggregator keys by `dongle_id`
today; if cars-per-dongle turns out to matter in the real fleet, swap that
key for `(dongle_id, car_vin)` in `fleet_lateral_aggregate.py`.

## Design choices, and why

- **Sufficient statistics, never timelines.**  Per-route output is ~50 KB
  regardless of route length; aggregation is `O(routes)` memory.
- **Three truth sources side-by-side.**  Prior work showed they disagree
  (R² ≈ 0.1 between QFK and yaw-rate derived).  Reporting all three keeps
  honest about which signal is being closed on.
- **Calibrated CC pose preferred over raw livePose.**  `controlsd` sets
  `CC.orientationNED` and `CC.angularVelocity` from `PoseCalibrator` —
  i.e. mounting tilt already subtracted.  Using these avoids reimplementing
  the calibrator and removes the device-tilt failure mode that the prior
  scripts hit (raw livePose roll has ±5° tilt that fails sunnypilot's 0.10
  m/s² roll gate on most highway samples).
- **`hca_sent` from `carOutput.actuatorsOutput.curvature`**, not CAN.  This
  is what the controller actually wrote to the wire, including all
  slew-limit / safety post-processing.  CAN HCA_03 would tell the same
  story plus rack-side post-processing noise.
- **Engagement gate** uses 2.0 s buffer after `latActive` rising edge and
  after the last `steeringPressed[Slightly]` event — matches sunnypilot's
  learner so results are comparable to it.
- **Lat-delay alignment** via `liveDelay.lateralDelay`: at sample time t,
  compare `truth(t)` to `desired(t − lateralDelay)`.  Without this,
  transient curvature changes contaminate steady-state gain estimates.

## Known caveats

- `carState.yawRate` is not populated on most VW carstate branches.  The
  `cs_yaw` slot becomes empty for those routes — gain stats for `cs_yaw`
  should be ignored on VW analyses.
- The slowest single-route runs in the smoke (≈ 200 s) were dominated by
  LogReader's initial segment download.  Cached re-runs drop to ≈ 20 s.
  For the real fleet pass, expect the first sweep to be I/O-bound on
  download bandwidth.
