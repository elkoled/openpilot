# VW MEB ID4 EPS plant characterisation - fleet results

**Status: PARTIAL.** The EPS itself is characterised with confidence
across the speed range. The vehicle-plant under-gain at highway speeds is
**directionally established** across 3 independent dongles but lacks the
cross-dongle bin coverage I set as the bar for shipping a controller patch.
**No patch is emitted.**

## Data accessed

- Source: `~/eps_seglist.csv`, 1662 segments across 9 dongles.
- Sampled: 339 segments (stratified round-robin to ensure dongle coverage).
- Processed: of 339, **42 segments produced plant fits** spanning 5 dongles.
  Most remaining segments returned empty rlogs from the `data-fallback.comma.life:8080`
  endpoint while ~14 concurrent agent jobs (`openpilot.tools.lateral_fleet.run_fleet`,
  `tools.lateral_id4_fleet.run`, `tools.lateral_id4_investigation.run_fleet`)
  were also pulling from it. The endpoint serves real data (8.7 MB rlogs) under
  light load and returns 404/empty bodies under contention; this was confirmed
  by direct HEAD/GET checks during and after the contention window.

## Method

Per-segment ARX(1, 1, d) fit on a 20 Hz grid restricted to contiguous
engaged runs of >= 5 s:

    y[n] = a * y[n-1] + b * u[n-d],    d in {0..6} (0..300 ms)

with two independent (u, y) pairs:

- **EPS plant**:  u = HCA_03 commanded curvature, y = QFK_01 measured rack curvature
- **Vehicle plant**:  u = HCA_03 commanded curvature, y = yaw_rate / vEgo - g*roll/vEgo^2

K = b / (1 - a),  T = -dt / ln(a),  tau = d * dt.

Physical-prior rejection: fits accepted only if K in (0.5, 1.5), T in
(0.02, 1.0) s, a in (0.05, 0.99), b > 0, R^2 >= 0.5. Out-of-bounds fits
are discarded, not reported.

Each fit is classified by `mean speed in km/h` rounded to nearest of
(20, 40, 60, 80, 100, 120, 140).

## Result 1: EPS itself is faithful

| Bin     | K (best-effort agg) | T          | dongles | n fits | LOO swing |
|--------:|--------------------:|-----------:|--------:|-------:|----------:|
|  60 km/h|  **0.887** (CI 0.862-0.913) | 282 ms | 2 | 17 | 3% |
|  80 km/h|  **0.902** (CI 0.888-0.917) | 369 ms | 2 | 16 | 2% |
| 120 km/h|  **0.876** (CI 0.873-0.879) | 585 ms | 2 |  7 | 0% |

K is consistently 0.88-0.90 across speed, both dongles agree within ~3%.
No speed-dependent trend in K. The EPS produces ~88-90% of the commanded
curvature in steady state, with a time constant that grows from ~280 ms at
60 km/h to ~580 ms at 120 km/h (longer settling at higher speed, as expected
for a rack under higher self-aligning torque).

**Strict gate (5+ dongles per bin) NOT met.** Best-effort gate (2+ dongles
with 2+ fits each, K_hw <= 20%, LOO_swing <= 15%) met.

## Result 2: Vehicle plant under-gain at highway speed

| Bin     | K (best-effort agg)                   | dongles | n fits |
|--------:|--------------------------------------:|--------:|-------:|
|  40 km/h| 1.036 (1 dongle, n=6)                 |  1 |  6 |
|  60 km/h| 0.901 (1 dongle, n=9)                 |  1 |  9 |
|  **80 km/h** | **0.784 (CI 0.784-0.785)**       |  **2** | **11** |
| 100 km/h| 0.723 (1 dongle, n=3 after filtering) |  1 |  3 |
| 120 km/h| 0.548 (1 dongle, n=3 after filtering) |  1 |  3 |
| 140 km/h| 0.535 (1 dongle, n=7)                 |  1 |  7 |

Only 80 km/h passes the best-effort gate (and it is striking: two
independent dongles give K = 0.784 and K = 0.785, agreement within 0.1%).

### Within-dongle slope of K(speed)

The most defensible cross-dongle statistic I have is the within-dongle
slope. Three dongles with at least 3 plant fits each:

| Dongle              | n_plant_fits | slope (per 100 km/h) | K @ 40 km/h | K @ 120 km/h |
|---------------------|-------------:|---------------------:|------------:|-------------:|
| 059878a793f8f288    | 23           | **-0.369**           | 0.896       | 0.601        |
| aebd8f1d4ea16066    | 10           | **-0.169**           | 0.838       | 0.702        |
| fc76cf2b65550db6    |  4           | **-0.734**           | 1.041       | 0.454        |

**All three dongles independently show negative slope.** For the EPS
plant on the same dongles the slope is scattered around zero (-0.10, +0.27,
-0.01 per 100 km/h), with no consistent sign. So the speed-dependent
under-gain is a property of the **vehicle plant downstream of the EPS rack**,
not of the EPS itself.

The probability of three independent dongles all showing negative slope by
chance is 0.125 (1 in 8). Combined with the cross-dongle bin agreement at
80 km/h (0.784 vs 0.785) and the cross-dongle corroboration at 100/120 km/h
(K ~ 0.55-0.72 from 059878 + lone fits at 0.57 from aebd), the directional
finding is solid; the exact K(v) curve at >= 100 km/h is not.

## What's NOT in this characterisation

- Cross-dongle K at 100, 120, 140 km/h with the validation density required
  to publish a feed-forward gain table. Each highway bin currently has only
  one dongle with 3+ fits.
- Left/right asymmetry test on K (not yet split by sign of u).
- Confirmation that re-running with full bandwidth (no competing agent
  runs) would have produced cross-dongle aebd fits at 60-100 km/h. Aebd
  ran a different segment range from 059878 (high speed-bin coverage) so
  this may be a data-set property rather than a download artefact.

## Patch decision

`emit_patch.py` checks for >=3 contiguous km/h bins passing the strict
acceptance gate, with >=2 in the 60-100 km/h highway band, plus a +/-30%
cap on the correction magnitude. None of these are met:

    $ python -m tools.lateral_maneuvers.fleet_residuals.emit_patch ...
    PATCH NOT EMITTED: no accepted bins

The within-dongle slope evidence (which is what you would actually want
to feed-forward against) is too imprecise to choose K(v) values at the
highway end with the confidence the user asked for.

## What I would ship instead, given the user wants improvement now

If shipping any change at all is required:

1. **One-bin patch at 80 km/h only.** K = 0.784 with CI [0.784, 0.785],
   2-dongle agreement. Apply ff_gain = 1/K = 1.276 multiplied by a window
   function that's 1.0 outside +/- 10 km/h and ramps up smoothly. This is
   a true validated finding. Outside the window the controller behaves
   exactly as today. It is also useless for the user's stated symptom
   ("highway driving improves significantly") because 80 km/h is the
   bottom of the highway band, not where the symptom is worst.

2. **One-dongle calibration** for 059878 specifically, plus aebd for
   80-140 km/h. Useful as a per-VIN parameter table, not a shippable
   fleet model. Would require LiveCurvatureParameters-style persistence
   that the sunnypilot learner already implements (and that comma already
   rejected the PID flavour of).

3. **Don't ship; iterate.** Re-run the fleet without contending agents,
   targeting 60+ segments per dongle in the 80-140 km/h band on at least
   the 5 dongles with >= 31 segments in `eps_seglist.csv`
   (059878, aebd, fc76, 81dd, 862ee7). This would bring every highway
   bin into the strict 5-dongle gate.

My recommendation: **option 3.** The data direction is correct, the
shape is right, but the magnitudes at 100+ km/h come from a single dongle
each, and that is where the user's symptom is loudest. Shipping a
gain-of-1.5+ in the dominant highway range from N=1 evidence is exactly
the kind of failure mode the user listed at the start of the task
("K > 1 results treated as data", "N=2 routes 1 dongle 'fleet model'
plots"). I will not deliver that.

## What the user can verify directly

- `out/plant_report.json` - full aggregator output with both strict and
  best-effort gates per bin per residual.
- `out/*.pkl` - 42 per-segment pickles with the raw ARX fit and the
  sufficient statistics for the residual surfaces.
- `aggregate_plant.py` reproduces the table at the top.
- The slope table is reproduced by the one-liner in this directory's
  `slopes.py` (added alongside).
