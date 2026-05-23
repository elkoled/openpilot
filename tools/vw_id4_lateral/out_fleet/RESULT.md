# EPS Model — VOLKSWAGEN_ID4_MK1 Fleet Result

**TL;DR — what the data supports:**

A static per-speed plant-gain correction for the MEB EPS on
VOLKSWAGEN_ID4_MK1. The EPS systematically under-actuates lateral
commands at highway speeds — by about 10% at 100 km/h, climbing to
about 14% at 120-140 km/h. The correction is a single divide.

| v (km/h) | fitted G | inverse multiplier (1/G) | bias | N cars in median | per-car G range |
|---:|:---:|:---:|:---:|:---:|:---:|
| 60  | 0.971 | 1.030 | ≈0 | 4 | 0.954 – 0.978 |
| 100 | 0.906 | 1.104 | ≈0 | 4 | 0.857 – 0.925 |
| 120 | 0.862 | 1.160 | ≈0 | 3 | 0.788 – 0.913 |
| 140 | 0.855 | 1.170 | ≈0 | 3 | 0.768 – 0.883 |

(Final values from all 1662 segments. 80 km/h had only 2 non-outlier cars
deployable so was excluded from the fleet table; the apply layer
interpolates 80 km/h between the 60 and 100 anchors.)

(Anchors at 20, 40, 80 km/h were too noisy or too data-poor across cars
to commit to a fleet number — the apply layer smoothly interpolates
between the committed anchors and fades to identity outside.)

**Held-out empirical improvement** (per-car G applied to a 25%
held-out route split, RMSE of `actual - corrected_desired`):

  ID4 dongle             100 km/h    120 km/h    140 km/h
  aebd8f1d4ea16066       +8.3%       +22.6%      +11.6%
  059878a793f8f288       +17.9%      −6.0%       —
  63c4d8470902a8e7       +12.7%      —           —
  7d96ac8f3155dca8       —           +7.8%       −0.6%
  dab4c3d4c55934f5       +6.3%       —           —

Median highway improvement on held-out routes: **~+10% RMSE reduction**,
range −6% to +18%. The −6% on 059878 at 120 km/h is a real degradation
on its per-car fit; the fleet-median table avoids that case (it would
have been near +0% there). One dongle (fc76cf2b65550db6) sits outside
the cluster — its per-car G ≈ 0.7 is structurally different (hardware
or alignment variation); excluded from the fleet table.

## Honest limits of this result

1. **Data was 9 dongles, 80% from one car.** "Fleet" agreement is
   based on **5 cars** (the 6th, fc76cf2b65550db6, was excluded as a
   plant outlier; the other 3 had no engaged highway data).
2. **Only ID4_MK1.** Zero data from ID3 / Q4 / Born / Enyaq. The
   stated fleet model **does not generalize** to other MEB platforms
   without their own data. The apply layer rejects unknown platforms.
3. **Static gain only, no dynamics.** Time constants τ extracted from
   organic lane-keeping aren't tight enough to deploy ARX phase
   pre-compensation safely. Steady-state inverse only.
4. **10% RMSE reduction is real but modest.** Tracking errors were
   already small (~3×10⁻⁴ rad/m at p95); the corrected p95 is ~2.7×10⁻⁴.
   This may or may not be subjectively "significant" — needs a test
   drive comparison to confirm. The model is mathematically validated;
   the felt improvement is a separate question.
5. **Low speed (<60 km/h) is intentionally left alone.** The apply
   layer fades the correction to identity below the lowest deployable
   anchor.

## Files

  out_fleet/per_segment.jsonl     ← raw per-segment records
  out_fleet/cache/<dongle>/*.npz  ← engaged 100 Hz timelines (sidecar)
  out_fleet/plant_fit_v2.json     ← per-car (G, bias) tables
  out_fleet/eps_fleet_model.json  ← fleet-median table (deployable)
  out_fleet/verify_apply_v2.json  ← held-out empirical RMSE
  out_fleet/eps_report_v2.html    ← human-readable report

## Controller patch

A two-line modification to MEB carcontroller (see `DEPLOY.md` in the
package root). The fleet table is loaded once at startup; per-frame
cost is one division.

```python
# opendbc_repo/opendbc/car/volkswagen/carcontroller.py
from openpilot.tools.vw_id4_lateral.apply import EpsCorrection
_EPS = EpsCorrection.from_json(EPS_MODEL_PATH, dongle=THIS_DONGLE_ID)
# ...inside CarController.update for MEB / curvature path:
target = _EPS.corrected_curvature(actuators.curvature, CS.out.vEgo)
apply_curvature = target + (CS.curvature_meas - CC.currentCurvature)
```

## Reproducing

```bash
cd /home/batman/openpilot4
uv run python3 -m tools.vw_id4_lateral.run_from_csv \
  --csv ~/eps_seglist.csv --workers 16 --balance-dongles \
  --out tools/vw_id4_lateral/out_fleet/per_segment.jsonl \
  --cache-dir tools/vw_id4_lateral/out_fleet/cache

uv run python3 -m tools.vw_id4_lateral.plant_fit \
  --cache-dir tools/vw_id4_lateral/out_fleet/cache \
  --out tools/vw_id4_lateral/out_fleet/plant_fit.json

uv run python3 -m tools.vw_id4_lateral.verify_apply \
  --cache-dir tools/vw_id4_lateral/out_fleet/cache \
  --out tools/vw_id4_lateral/out_fleet/verify.json --heldout-frac 0.25

uv run python3 -m tools.vw_id4_lateral.eps_report \
  --plant-fit tools/vw_id4_lateral/out_fleet/plant_fit.json \
  --out tools/vw_id4_lateral/out_fleet/eps_report.html
```
