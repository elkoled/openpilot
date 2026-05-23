# Deploying the fitted EPS correction

The fitted plant model is a per-(dongle, speed bin) static gain `G` and bias,
delivered as `plant_fit.json` and consumed by `apply.py:EpsCorrection`.

## Integration into the MEB carcontroller

The current MEB carcontroller line
(`opendbc_repo/opendbc/car/volkswagen/carcontroller.py:82`) is:

```python
apply_curvature = actuators.curvature + (CS.curvature_meas - CC.currentCurvature)
```

To deploy a per-car correction, insert ONE call before that line and use the
corrected curvature as the steady-state target:

```python
# at module top
from openpilot.tools.vw_id4_lateral.apply import EpsCorrection
_EPS_CORRECTION = EpsCorrection.from_json(
    "/data/openpilot/eps_plant_fit.json", dongle=os.environ.get("DONGLE_ID"))

# inside CarController.update, replacing line 82:
target_curvature = _EPS_CORRECTION.corrected_curvature(actuators.curvature, CS.out.vEgo)
apply_curvature = target_curvature + (CS.curvature_meas - CC.currentCurvature)
```

The correction class:
- Returns identity (passthrough) for non-deployable speed bins.
- Smoothsteps to identity below 20 km/h and above 140 km/h.
- Bounded G ∈ (0.5, 1.5), bias clipped to ±5e-4 rad/m. No code path can
  produce a curvature outside the existing
  `CCP.CURVATURE_LIMITS.CURVATURE_MAX = 0.195` envelope because the
  downstream `apply_std_curvature_limits` is unchanged.

## Per-VIN selection

`plant_fit.json` is keyed by dongle. For a multi-vehicle fleet you would
either:

1. Ship one `plant_fit.json` per car and key by VIN at boot (preferred),
2. Or ship a single ID4_MK1-wide table built from the deployable cells
   that AGREE across cars (where the per-car LOO-CV agreement is high) and
   pass through for cells where they disagree.

Option 2 produces a single ship-once correction. Option 1 produces
per-car corrections with higher confidence but more deployment surface.

## What the model does NOT cover

- **Dynamic phase compensation.** The static inverse handles steady-state
  gain and bias only. We do NOT deploy the ARX(1,1,Td) phase-lead
  correction because the time-constant estimates from organic
  lane-keeping data are not tight enough to deploy without risk of
  oscillation.
- **Cars not represented in the fit.** Only the dongles in the input CSV
  have their own correction. For an ID4_MK1 not in the dataset, the
  correction passes through unchanged (G=1, bias=0).
- **Platforms other than ID4_MK1.** The fit has zero MQB / MLB / ID3 / Q4 /
  Born / Enyaq data. The correction class refuses to apply to unknown
  fingerprints — caller responsibility.

## Pre-deployment checklist (do these before flashing)

1. Verify the dongle's VIN/fingerprint matches an entry in
   `plant_fit.json`.
2. Inspect the per-speed `(G, bias)` table — values should monotonically
   change with speed (small steps) and stay inside the bounds.
3. Run the held-out fold from `plant_fit.py --report` and confirm the
   per-route LOO RMSE is < identity RMSE × 0.90.
4. Make a short test drive at 80–120 km/h with the correction off,
   record `(actuators.curvature, CS.curvature_meas, CS.vEgo)`. Repeat
   with the correction on. Compute `|c_meas - target|` p50 / p95 in both
   recordings — corrected should be lower by the LOO-predicted margin.

If step 4 disagrees with the LOO prediction, do NOT deploy.
