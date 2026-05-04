# VW MEB / ID.4 MK1 — Consolidated Port Reference

Conventions: numeric CAN addresses are decimal in DBC `BO_` headers, hex
everywhere else. "Bus 0 = Bus.pt = gateway/powertrain", "Bus 2 = Bus.cam =
camera/radar", "Bus 1 = Bus.alt = extended/radar (PreCrash_02 here)".

---

## 0. Validation routes

All on owner `aebd8f1d4ea16066`:

| Route segment | What it shows |
|---|---|
| `0000006c--6d07717278` | Stock ACC working baseline, 19 931 frames, 4162 latActive, ~2068 angle-saturated frames at v>5 |
| `000000d8--a443aceeb9` | Highway w/ overrides, 184 649 frames, sustained low-speed saturation |
| `000000d9--031829e8ca` | Tight low-speed curves, 15 749 frames, heavy sustained driver torque (mean 111 Nm) |
| `000000ed--1c9bdc8e9e` | Engage → drive ~25 s → gas-override → EPS goes FAULT then permanent-fault cycle. 12 695 carState frames; LKAS-fault repro |
| `000000f3--69d35c80a9` | Stock EA escalation, 386 s, 13 TA cycles (3 events with PHASE 2 + PHASE 3) — driver intervened before standstill |
| `000000f4--70de8333d7` | Stock EA escalation, 489 s, 14 TA cycles (3 events with PHASE 2 + PHASE 3) — driver intervened before standstill |

`f3` and `f4` are stock-mode (openpilot in dashcam) and are the canonical
**measured** EA escalation reference — everything theoretical anywhere else in
this doc bows to these traces. Replay tool: `tools/lib/logreader/LogReader('aebd8f1d4ea16066/<seg>/0/r')`.

Sample-rate caveat — **qlogs drop most CAN frames**. Verified: addresses
0x24C/0x24D/0x2B7 returned 0 hits on the q-cache for `000000ed`. EA_02 was
**completely missing** from the f3/f4 qlogs. For DM/AEB replication, **rlogs**
are mandatory (`rlog.bz2` from `/data/media/0/realdata/<route>/<seg>/`).

---

## 1. Carport status

### 1.1 Lateral (steering)

- **VM-style safety** (Tesla pattern) — `volkswagen_meb_curvature_to_angle()`
  converts the HCA_03 curvature payload into 0.1 deg of steering wheel via
  the bicycle model (steer_ratio=15.6, wheelbase=2.77, slip=−0.000605).
  `steer_angle_cmd_checks_vm` runs in angle space; rate limit dynamic via
  ISO accel + ISO jerk. `angle_deg_to_can = 10`. `max_angle = 6000`
  (600 deg, well above the rack lock-to-lock).
- Conversion path each TX:
  1. Read curvature from HCA_03 bytes 3-4 (15-bit, scale 6.7e-6 rad/m)
  2. `volkswagen_meb_curvature_to_angle(curv_can)` → steering wheel angle
     in 0.1 deg, applying the bicycle-model curvature factor at current speed
  3. `steer_angle_cmd_checks_vm` applies ISO accel + ISO jerk in angle
     space, plus inactive-near-meas check when `steer_req=False`
  4. Power check: `desired_power <= 125` and `desired_power == 0 when !steer_req`
- Carport mirror: `apply_std_curvature_limits` (or `apply_std_steer_angle_limits`)
  matches safety with `MAX_LATERAL_ACCEL = MAX_LATERAL_JERK = ISO + g·0.06 ≈ 3.59`
  in curvature space. Mathematically equivalent to safety's angle-space at
  slip=0; with non-zero slip the difference is <1 % at 40 m/s and the carport
  always sees a slightly tighter bound than safety.
- Safety vs carport alignment fuzz: 4500/4500 frames pass `tx_hook` across
  speed/curvature/measured-curvature/last-apply combos with
  must-be-physically-reachable-from-prior-frame filter applied.
- Closed-loop EPS feedforward: `target = actuators.curvature + (measured -
  currentCurvature)` → VM → rate-limit → VM back → safety clip.
- Power ramp: speed-aware floor (`STEERING_POWER_MIN=4`), driver-aware
  ceiling (`STEER_DRIVER_ALLOWANCE=60..150`, `STEER_DRIVER_MAX=300`), slew
  ±`STEERING_POWER_STEP=2` per send.
  - `target_power_driver = interp(|driver_torque|, [60..300], [50..4])` softly
    cuts power on driver pull
  - Inactive branch keeps `hca_enabled=True` while ramping power down to 0 —
    avoids HCA "REJECTED" on transitions
  - Speed-gating: at v<0.5 m/s the target is forced to MIN so the EPS isn't
    slammed at standstill
- Graceful disengage (op10-aligned): when `latActive` flips False, keep
  `hca_enabled=True` while `steering_power` ramps MAX→0 (-2 per HCA frame).
  Stops EPS PERMANENT FAULT on disengagement (route 000000ed reproduced
  the fault on the old code; line-for-line match to `update_meb` shutdown branch).
- HCA_03 inactive-curvature branch sends `clip(measured_curvature, ±0.195)`
  so safety's inactive-anchor (`angle_meas`) accepts.
- **Take-over alert path**: `update_meb` sends `LDW_02` with `LDW_Texte =
  laneAssistTakeOver(8)` whenever `hud_control.visualAlert in
  {steerRequired, ldw}`. Before this fix, carcontroller emitted no `LDW_02`
  at all — dashboard never lit the take-over chevron, no matter how
  saturated the model was.
- `steeringPressed` debounce: `update_steering_pressed(condition, 5)` =
  5-frame hysteresis (50 ms @ 100 Hz). Same pattern as
  Tesla/Ford/PSA/Hyundai/Rivian. With `STEER_DRIVER_ALLOWANCE` choices in
  the wild: 60 (op10 / current), 80 (some forks), 150 (op3 — 1.5 Nm, above
  route p95 road force ~1.36 Nm; below real driver pulls p99 ~3.09 Nm).
  Note: bumping to 150 was tried; 60 was retained per ground-truth alignment
  with op10. Bar flicker is now solved by the hysteresis instead of by
  raising the threshold.

### 1.2 Longitudinal

- ACC_18 envelope: `ACCEL_MIN=-3.5`, `ACCEL_MAX=2.0` m/s².
- **Override-release jerk smoothing** (current): rate-limit `accel` to
  ±2.0 m/s³ between ACC frames; hold `accel_last = 0.0` while override is
  active so the post-override ramp starts from 0 (what the car actually
  saw). Reaches full braking in ~1.5 s instead of one frame; no more snap
  from 0 → −3 m/s² when releasing gas behind a stopped lead. Variants
  tried elsewhere — op1 also rate-limits with `ACC_JERK_MAX=2.5`, op3 uses
  a release-only state machine at 2.5 m/s³ exiting on |Δ|<0.05, op4 uses
  1.0 m/s³ for first 50 frames then 3.0 thereafter (4.5 if FCW), op5 5-frame
  linear blend, op6 hardcoded 0.1 m/s²/frame ≈ 5 m/s³. **Op2's 7-line
  patch is what shipped** (smallest diff, no values.py change, no state
  machine).
- `acc_control_value` / `acc_hold_type` switch ACC_18 between
  active / override / stopping / hold / ramp_release states.
  - `acc_hold_type` op10-equivalent: RAMP_RELEASE on override-begin and on
    disengage so the EPB does not snap on/off; HOLD when stopping or
    `esp_hold`; RELEASE on starting; NO_REQUEST otherwise.
- `long_stopping_counter` keeps `stopping=True` for 25 frames after
  openpilot drops out of `LongCtrlState.stopping` while still at very low
  vEgo. Avoids flickering between stopping and non-stopping at the moment
  the cluster decides we have stopped vs not.
- `acc_hold` HMS state: sustained PHASE2 deceleration is delivered via
  ACC_18 + `TSK_Status=brake_only` on stock, NOT via `EA_Warnruckprofil`
  (which stayed 0 in every captured EA event — see §6).

### 1.3 Set-speed read-back

- `update_meb` reads `MEB_ACC_01.ACC_Wunschgeschw_02` (10-bit, 0.32 km/h
  scale, 0–327.04, 1023 = no display) when `pcmCruise` is set; subscribes
  the cam parser to `MEB_ACC_01` at 25 Hz. Sentinel >320 km/h → 0.
- With openpilot longitudinal we generate the message ourselves so the
  read is bypassed. Cluster always reads km/h.

### 1.4 HUD

- `LDW_02` (Lane Assist HUD) sent at `LDW_STEP=10` cadence. Fixes prior
  "Lane Assist not available" cluster message. Take-over text
  (`LDW_Texte=8`) wires `visualAlert in (steerRequired, ldw)`.
- `MEB_ACC_01` (cluster ACC widget) sent at `ACC_HUD_STEP=6` with
  `ACC_Wunschgeschw_02 = hud_control.setSpeed * KPH`.
- `display_mode = 1 if lat_active else 0` for travel-assist style yellow
  lanes, matches op10 line-for-line.

### 1.5 BSM (corner radars)

- `MEB_Side_Assist_01` (0x24C, 16 B, bus 2, 50 Hz) populates
  `Blind_Spot_Info_Left/Right` and `Blind_Spot_Warn_Left/Right`. Carstate
  reads from `cam_cp`. `enableBsm = 0x24c in fingerprint[2]`.
- **Bug fix this session**: `get_can_parsers_meb` was registering an empty
  cam-bus parser. `cam_cp.vl["MEB_Side_Assist_01"]` returned zeros silently
  → BSM was always False even when fingerprint enabled it. Now subscribes
  `("MEB_Side_Assist_01", 20)` when `CP.enableBsm`. End-to-end on heavy
  override route: `leftBlindspot=11 381`, `rightBlindspot=10 474` frames.

### 1.6 Front radar tracks

- `MEB_Distance_01` (0x24F, 64 B, bus 2, 25 Hz) parsed by
  `radar_interface.py`. 6 fused tracks (3 lanes × 2 slots) with stable
  ObjectIDs. Track ID 16 followed continuously 7+ s on route 000000ed.
- `Bus.radar: 'vw_meb'` added to `VolkswagenMEBPlatformConfig.dbc_dict`.
  `interface.py` MEB branch overrides `ret.radarUnavailable = False`.
- DBC scale note: this port `0.07 m / 0.065 m`; op10 scale `0.28 / -143.36`
  is the more recent revision.
- Op10 facelift renames `MEB_Distance_01 → Strukturen_01` (would break our
  parser when MK2 lands).

### 1.7 Carstate yawRate

- `ret.yawRate = -pt_cp.vl["ESC_50"]["Yaw_Rate"] * (1, -1)[Yaw_Rate_Sign] *
  CV.DEG_TO_RAD` — fixes undershoot detection in `_check_saturation` which
  uses `yawRate * vEgo` for actual lateral accel.

### 1.8 Fingerprint / chassis

- `VOLKSWAGEN_ID4_MK1` declares `chassis_codes={"E2"}` in `values.py`.
  Several captured routes are on **chassis E8** (US-built 21MY ID.4 from
  Chattanooga). **Action**: add `"E8"` to the set before merging — without
  it, US cars do not VIN-fingerprint.
- Missing WMIs vs op10: op10 ships `{USA_SUV, EUROPE_CAR, EUROPE_SUV}`,
  this port has `{EUROPE_SUV}` only — adding the 3 missing WMIs unblocks
  US-built and many EU-VIN ID.4s.
- Op10 also bundles ID.5 in the same CarDocs (same E2 chassis).
- Dashcam mode: removed for MEB ID.4 (this is a full stable port). All
  other ports use `ret.dashcamOnly = is_release` for provisional support;
  the field defaults to False per capnp schema.

### 1.9 Tests

- `opendbc/safety/tests/test_volkswagen_meb.py` — 63 tests, 100% coverage
  on `volkswagen_meb.h`. 3157 total safety tests, 100% coverage maintained.
- `opendbc/car/volkswagen/tests/test_meb_routes.py` (op3) — per-route
  regression: safety alignment, driver-allowance flicker, alert eligibility
  on saturated frames. Skips intelligently when route can't structurally
  trigger.
- `opendbc/car/volkswagen/tests/test_meb_alerts.py` (op2) — replays all 4
  user-provided routes through unmodified upstream
  `LatControlAngle._check_saturation`, locks expected `steerSaturated`
  firing ranges:
  - `0000006c` (stock, hands-off): 1500–3500 frames fire
  - `000000d9` (flicker / driver in curves): 0–100 frames (mostly suppressed
    by `steeringPressed` — correct)
  - `000000d8` (highway override, 124 k frames): 150–800 fire at real
    intervention moments
  - `000000ed` (EPS-fault drive): 0–50 frames (controller frequently disengaged)
  Gated by `REPLAY_ROUTES=1` (downloads qlogs).
- `TestVolkswagenMebTakeoverAlert` (op4) — 4 unit tests, 8 route-scenario
  subtests: saturated mid-curve, driver-pressed override, sustained tight
  curve, engage-with-active-alert, plain LDW, normal driving,
  disengaged-no-alert, disengaged-with-alert. Asserts `LDW_Texte` matches
  the DBC laneAssistTakeOver slot (8) when expected, and that the
  green/yellow status LEDs track `latActive` and `steeringPressed`.
- 247 car interface tests pass; 162 existing VW family tests
  (MQB/MLB/PQ) — no regressions.

### 1.10 Take-over alert / steerSaturated wiring (root-cause fix log)

- Replicated user-reported failures on route `000000ed`:
  - "LKAS Fault: Restart the car" fired ~47.4 % of frames (false positive)
  - Take-over alert never fired
- Root cause: `update_hca_state` (shared MQB/MLB helper) treats EPS
  `FAULT` as permanent. ID.4 EPS reports `FAULT` whenever the driver
  applies sustained override torque, then auto-recovers.
- Fix iterations attempted:
  - inline MEB-specific logic in `carstate.update_meb` (ALL of FAULT/
    REJECTED/PREEMPTED → temporary; perm = False)
  - 2-second debounce on FAULT in `update_hca_state` helper
  - **Final**: aligned with op10 verbatim — `update_hca_state` shared
    helper, MEB DISABLED + sustained FAULT after `eps_init_complete` =
    permanent; REJECTED/PREEMPTED = temporary. No debounce.
- After fix: 0 % permanent alerts on the route, 49.9 % temp alerts
  (correctly tracks EPS state).


---

## 2. Set-speed — three usable paths

| # | Path | Status | Notes |
|---|------|--------|-------|
| A | `GRA_ACC_01` button spam | half-wired today; full impl in op3 | `mebcan.create_acc_buttons_control` only sends cancel/resume currently. Op10 / op3 patch is ~5 lines to add up/down. DBC has set/+/-/long-press/gap/main/limiter buttons + 4-bit COUNTER. Sunnypilot ICBM rate-limits 200 ms. Works under `pcmCruise`. |
| B | Direct `ACC_Wunschgeschw_02` | done | 10-bit, factor 0.32 km/h, 0–327.04, 1023=hide. `mebcan.create_acc_hud_control` writes it from `hud_control.setSpeed * MS_TO_KPH`. Cluster always reads km/h. Active when `openpilotLongitudinalControl=True`. Note: `MEB_ACC_01.ACC_Wunschgeschw_02` is **display-only** — TSK ignores it for the actual setpoint. |
| C | mph/kmh + cluster offset | not done | On-the-wire unit signal: `KBI_MFA_v_Einheit_02` (BO 1603, 1-bit), `TSK_Einheit_vMax_FahrerInfo` (BO 706). Clusters typically read 2–5 % high vs `vEgo`; needs bench data to derive offset. Op10 doesn't do this either. |

Skip indefinitely: `speed_limit_manager.py` (PSD predictive maps not
guaranteed on ID.4), `MEB_PACC_01` (predictive ACC, undocumented signals),
sunnypilot ICBM (pulls sunnypilot capnp schemas).

### Stalk inputs — `GRA_ACC_01` (msg 0x12B = 299, 33 Hz, bus 0)

| Signal | Bit | Effect |
|---|---|---|
| `GRA_Hauptschalter`        | 12 | ACC main on/off |
| `GRA_Abbrechen`            | 13 | CANCEL |
| `GRA_Limiter`              | 15 | toggle ISA / speed limiter mode |
| `GRA_Tip_Setzen`           | 16 | SET (one-shot tap → setpoint = current speed). Sunnypilot: `GRA_Tip_Setzen=1` alone → setpoint **−1 km/h**. Hold ≥0.6 s → step 5 km/h |
| `GRA_Tip_Hoch`             | 17 | speed +1 km/h (or +5/+10 with `Tip_Stufe_2`) |
| `GRA_Tip_Runter`           | 18 | speed −1 km/h |
| `GRA_Tip_Wiederaufnahme`   | 19 | RESUME. Alone: setpoint **+1 km/h** |
| `GRA_Verstellung_Zeitluecke` | 20–21 | gap up/down |
| `GRA_Tip_Stufe_2`          | 26 | "+5/+10" modifier (long-press) |
| `GRA_TravelAssist`         | 30 | Travel Assist enable |

Note: `GRA_Tip_Hoch` / `Tip_Runter` are not used by ID.4 stalks (PQ-era);
the MEB stalks use `Tip_Setzen` / `Tip_Wiederaufnahme` for ±1 km/h.

### Implementation sketch (op3 reference, ~30 lines)

1. Extend `mebcan.create_acc_buttons_control` with `up`/`down` kwargs that
   drive the same Wiederaufnahme/Setzen bits (or `Tip_Hoch`/`Tip_Runter`).
2. In `carcontroller.update_meb` translate
   `hud_control.setSpeed - CS.cruiseState.speed` into rate-limited presses
   (skip frames where COUNTER hasn't advanced; max 1 press per 200 ms / ~5 Hz).
3. Read cluster set-speed via `MEB_ACC_01.ACC_Wunschgeschw_02` to close
   the loop.
4. Why button-spam and not direct setpoint: `MEB_ACC_01.ACC_Wunschgeschw_02`
   is display-only (TSK ignores). `ACC_18` has no setpoint signal. TSK
   reads from cluster's internal cruise state, itself driven by GRA_ACC_01
   button presses.

### Risks (sunnypilot/op10 experience)

- Counter/CRC mismatch → cluster throws "Driver Assist Unavailable" + locks out.
- Sending too fast (>10 Hz) → cluster ignores presses.
- Sending `Setzen` while ACC is OFF can engage cruise unexpectedly.

### Effort estimate

~80 lines for full state machine: set_speed_target field, read-cluster +
compute-delta + send-Up/Down at 5 Hz + debounce, safety TX-side guard so
button taps only happen with ACC armed, tests for rate limiter.

---

## 3. Side radars — current + RE plans

### Hardware

Two **Continental SRR520** corner radars (24 GHz short-range), one per
rear corner. Different from MQB Side Assist (older SRR308 / Hella). Not
raw per-target track sources on the consumer-exposed CAN.

### Bus location

Extended/Komfort CAN, bridged through gateway. On most ID.4 installs
they appear on `Bus.cam` (via the camera relay).

### Message inventory

| Address | Name | Size | Rate | Decoded | Notes |
|---|---|---|---|---|---|
| 588 / 0x24C | `MEB_Side_Assist_01` | 16 B | 50 Hz (op6) / 20 Hz (op1) / 25 Hz (op4) | booleans | `Blind_Spot_Info_*` and `Blind_Spot_Warn_*` already used. Plus 7-bit `Blind_Spot_Right/Left` field with undocumented semantics (range 0–15). Also `Lower_Speed_01/02` / `Higher_Speed_01/02` overtaking-vehicle flags, `NEW_SIGNAL_1/_6` unidentified |
| 589 / 0x24D | `MEB_Side_Assist_02` | 64 B | 20 Hz | none | DBC has only two 3-bit unknowns. Empirical: ~25 of 64 bytes vary, two parallel ~24-byte regions (bytes 8-23 and 48-55) flip between FF padding and structured values — looks like two side-radar tracks but bit layout unknown. Likely internal track list (~10 tracks × 6 bytes) |
| 695 / 0x2B7 | `RCTA_01` | 8 B | 20 Hz | CRC + counter only | Payload bytes 2-7 constant on test routes. Likely emits track data only when reverse + cross-traffic present |

### Already implemented (this PR)

- `interface.py:49`: `enableBsm = 0x24c in fingerprint[2]` ✓
- `carstate.py:327-329`: ORs `Info|Warn` into `leftBlindspot` /
  `rightBlindspot` ✓
- `get_can_parsers_meb`: subscribes `MEB_Side_Assist_01` on cam bus when
  `enableBsm` set ✓ (this PR — without this the existing read path would
  KeyError)
- Live counts validated on `000000d8` (124 k frames): 1 left + 1 right
  rising edge while moving; right event had intensity `77/127` — strong
  proximity reading.

### Validation table

| Route | Frames | Info L / Warn L | Info R / Warn R |
|---|---|---|---|
| stock-acc-working | 19 931 | 300 / 300 | 300 / 300 |
| flicker | 15 749 | 300 / 300 | 300 / 300 |
| heavy-override | 184 649 | 11 306 / 354 | 10 399 / 354 |

End-to-end through `CarState.update_meb` with `enableBsm=True` on
heavy-override: `leftBlindspot=11 381`, `rightBlindspot=10 474` frames.

No safety-side change needed — BSM is informational only.

### To unlock raw side-radar tracks

Two RE datasets needed:
1. Steady drive with vehicles passing in adjacent lanes at known geometry
   (for `MEB_Side_Assist_02`).
2. Reverse out of a parking spot with crossing traffic (for `RCTA_01`).

Then bit-field-mine 589/695 against ground truth. Plot bit-rates per byte
in cabana — moving multi-byte fields surface as smooth signals.
Cross-reference timing with stock BSM LED activations.

Disambiguate trim presence via VCDS module `1C` PR codes (`6I3` / `6I7`).
Several hours of cabana work, not doable from qlogs.

---

## 4. Front-radar tracks (`MEB_Distance_01`)

`0x24F` broadcasts on bus 2 at 25 Hz with **6 object slots** ×
{ObjectID (6-bit), Long_Distance (0.07 m), Lat_Distance (0.065 m),
Rel_Velo}. Note scale difference: this port `0.25, -128`; op10
`0.28, -143.36`. **Op10's scale is the more recent one.** Live data
confirmed populated with leads.

Op10 DBC has additional per-slot `Same_Lane_01_Unknown_01..05` after
Right_Lane_02 — likely Distance_Status / quality / class but un-decoded.

### Activation path (ports 1, op10) — ~30 lines

1. Add `Bus.radar: 'vw_meb'` to `VolkswagenMEBPlatformConfig.dbc_dict` ✓
2. Port op10's `radar_interface.py` (parses 0x24F on bus 2 at 25 Hz, emits
   `RadarData.RadarPoint` per non-zero ObjectID) ✓
3. `interface.py` MEB branch override `ret.radarUnavailable = False` ✓
4. Optional: rename `MEB_Distance_01 → Strukturen_01` to match op10.

### Dedupe behavior

MEB publishes the same fused object on multiple lane slots (e.g.
Same_Lane_01 + Same_Lane_02 holding lead + next vehicle). Op1 silently
keeps first occurrence; op10 errors via `ret.errors.canError = True`.
Op2 wrongly used `Bus.pt` (bus 1) — rejected; op1's `Bus.radar`/bus-2
choice is correct. Adopted op1's silent dedupe with comment explaining.

### Auto-detect

`interface.py` auto-detects radar via `0x24F in fingerprint[0]`.

### `MEB_Distance_01` field map (per slot, 6 slots × 4 fields)

```
{Same|Left|Right}_Lane_{01|02}_ObjectID         6-bit
{Same|Left|Right}_Lane_{01|02}_Long_Distance    14-bit, 0.07 m
{Same|Left|Right}_Lane_{01|02}_Lat_Distance     10-bit, 0.065 m
{Same|Left|Right}_Lane_{01|02}_Rel_Velo         13-bit, 0.04 m/s
```


---

## 5. Trims, generations, facelift, MK1 vs MK2

### 5.1 ID.4 generations

| Year | Marketing | Chassis | DBC | Flag | Notes |
|---|---|---|---|---|---|
| 2021–2023 | ID.4 MK1 | E2 (some E8 US-built) | `vw_meb` | `MEB` | this port |
| 2023.5 EU | ID.4 22.5 facelift | E2 | `vw_meb` | `MEB` | Same DBC. `Motor_54.Accelerator_Pressure` becomes unreliable — already prefer `Motor_51.Accel_Pedal_Pressure` |
| 2024–25 | ID.4 MK2 / GEN2 (US "facelift") | **E8** | **`vw_meb_2024`** (sunnypilot only) | `MEB` + **`MEB_GEN2`** | Distinct DBC (ESC_51 48→64 B, Motor_51 32→48 B, MEB_Camera_09 8→3 B). HCA_03 / ACC_18 / QFK_01 / LWI_01 unchanged → carport logic carries over |

Note the ambiguity: meb6 says **no MEB MK2 ID.4 yet (as of 2026-05)** — VW's
next-gen ID.4 successor is on the **SSP (Scalable Systems Platform)** for
~2026/2027 with a completely new EE arch / zonal controllers; this DBC will
not apply. Other docs (meb1, meb3, meb4, meb5) describe MK2 as a real
facelift with the GEN2 deltas. Treat MK2 status as **uncertain** pending a
real VIN sample.

### 5.2 ID.4 trims (CAN-relevant deltas)

| Trim | Drivetrain | Battery | Mass (kg) | SteerRatio | CAN impact |
|---|---|---|---|---|---|
| Pure / Pure Performance | RWD 109–125 kW | 52 kWh | ~2000 | 15.6 | none. Often no SWA |
| Pro / Pro Performance | RWD 128–150 kW | 77 kWh | ~2150 | 15.6 | SWA optional |
| Pro S / Pro 4Motion / GTX 220 | RWD or AWD 195–220 kW | 77 kWh | ~2270 | 15.6 | second motor on front axle, EML_06 reports it. SWA standard |
| GTX 250 (late MY23) | AWD 250 kW | 77/82 kWh | ~2280 | 15.4 | sport-tuned EPS |
| GTX (general) | AWD dual-motor | 77/82 kWh | — | — | adds front motor + sport in Charisma; lateral/long signals identical |

All MK1 trims share EPS / radar / ESC firmware family.
`VolkswagenCarSpecs(mass=2224, wheelbase=2.77, steerRatio=15.6)` is set per
the GTX-laden upper bound; calc_slip_factor = −0.000605 (verified matches
safety constant).

### 5.3 Equipment packs that gate message presence (use for fingerprinting)

| Pack | Adds | Fingerprint |
|---|---|---|
| Side Assist | `MEB_Side_Assist_01/02` (588/589) | `0x24c in fp[2]` ✓ |
| Travel Assist | `TA_01` (619) | required for OP |
| Park Assist Plus | `Parken_SM_03`, `APS_Master` (850, 896) | irrelevant to OP |
| AEB / FCW | `AWV_03` (219), `VMM_02` (313), `PreCrash_02` (319) | always present on MEB |
| Capacitive wheel (KLR) | `KLR_01` (605) populated `KLR_Touchintensitaet_*` | facelift / equipped wheel. **Not detected today** — `0x25D in fingerprint` would do it |

### 5.4 ID.4 trim — feature ↔ openpilot-relevant detection

| Feature | Availability | How we detect |
|---|---|---|
| Travel Assist / IQ.Drive | All US trims, EU Pro+ standard, EU Pure optional | Required for openpilot — radar + camera FW present |
| **Side Assist** (BSM) | Std on Pro S / GTX / AWD Pro S; optional below | `0x24c in fingerprint[2]` (used in `interface.py`) |
| Front Radar (ARS5-B) | Same across trims with ACC; not on EU Pure base | `0x24F in fingerprint[0]` (now used by `radar_interface.py`) |
| Park Distance Control | Standard Pro+; optional below | not consumed |
| Area View / 360 camera | Pro S / GTX option | not consumed |
| Capacitive Wheel (KLR_01) | Standard MY24+ Pro S/GTX; optional pre-facelift | not detected yet |
| RCTA | Bundled with Side Assist | `RCTA_01 (0x2B7)` — only header decoded |

US-market quirk: nearly every US ID.4 ships with IQ.Drive; **side radars
are the wildcard**.

### 5.5 Trim-relevant carport flags (ports and op10 flag list)

| Trim flag | Effect |
|---|---|
| `STOCK_HCA_PRESENT` | always on Pro S+; gates EA torque-spoof in carcontroller |
| `STOCK_KLR_PRESENT` (op10 = 64) | car has Travel Assist 3.0 capacitive wheel; gates the KLR_01 hands-on spoof. Without spoofing, openpilot's torque alone won't satisfy FAS; chime fires. Without KLR, older torque-only hands-on detection — `STOCK_HCA_PRESENT` torque spoof handles it |
| `KOMBI_PRESENT` | dashboard variant detect |
| `STOCK_PSD_PRESENT` | predictive map data ECU present |

### 5.6 Op10 chassis variants we don't have

| Platform | Chassis | Mass | Wheelbase | sR | Flags |
|---|---|---|---|---|---|
| **VOLKSWAGEN_ID3_MK1** | E1 | 1935 | 2.77 | 15.6 | MEB |
| **VOLKSWAGEN_ID4_MK2** | E8 | 2224 | 2.77 | 15.6 | MEB + MEB_GEN2 |
| **VOLKSWAGEN_ID3_MK2** | E1 (year R/S) | 1935 | 2.77 | 15.6 | MEB + MEB_GEN2 |
| AUDI_Q4_MK1/MK2 | FZ | 1965 | 2.764 | 15.6 | MEB / MEB+GEN2 |
| SKODA_ENYAQ_MK1/MK2 | NY (or 5A / OX) | 1965 | 2.77 | 15.6 | MEB / MEB+GEN2 |
| CUPRA_BORN_MK1 | K1 (or KL) | 1950 | 2.766 | **15.9** | MEB |
| FORD_EXPLORER_EV_MK1 | EF | 2090 | 2.77 | 21.7 | MEB + MEB_GEN2 |

MK1 vs MK2 split is by **VIN year letter** (`vis[0]`): M/N/P → MK1, R/S → MK2.
Op10 added a `model_years` field on `VolkswagenMEBPlatformConfig` plus a
third axis to `match_fw_to_car_fuzzy` (values.py:726, 747).

### 5.7 MEB chassis variants — full table

| Code | Vehicle | Status |
|---|---|---|
| E1 / E11 | VW ID.3 MK1/MK2 | Same EPS limits as ID.4 (validated in sunnypilot) |
| E2 / E21 | VW ID.4 MK1 | this port |
| E22 | VW ID.5 (sister coupé MK1) | Shares ID.4 stack |
| E23 / E26 | VW ID.6 (China) | Untested |
| E3 | VW ID.7 | sedan / wagon, GEN2 from launch |
| E8 | VW ID.4 MK2 (US-built) | needs `MEB_GEN2` |
| EB1 | VW ID.Buzz (MEB-LWB) | Longer wheelbase changes lateral tune `[uncertain]` |
| F4 / FY / FZ | Audi Q4 e-tron / Q4 Sportback / Q4 MK2 | premium sibling, slight ACC retune |
| K1 / KL | Cupra Born / Cupra Tavascan | K1 = ID.3 hot-hatch, KL = ID.5 twin |
| 5A / NY / OX | Škoda Enyaq iV / Coupe MK1/MK2 | Largest non-ID.4 validation base (jyoung8607) |
| EF / EH | Ford Explorer EV / Capri EV MK1 | licensed MEB |

Per sibling, only `VolkswagenCarSpecs(...)` and the safety
AngleSteeringParams need updating; the EPS message format is shared.

Detection via `FW_VERSIONS` ECU part numbers (e.g. `1EA909144*` = ID.3 EPS,
`11A909144*` = ID.4 EPS, `5LA909144*` = Enyaq EPS). Use `FW_VERSIONS` in
`fingerprints.py` rather than CAN message presence for trim-level detect.

### 5.8 Year-to-year fingerprints

The single MK1 fingerprint set covers any 2021–23 ID.4 — no clean year
sub-clustering. The meaningful boundary is MK1 vs MK2:

- chassis: E2 (MK1) vs E8 (MK2)
- fwdRadar part: `1EA907572*` (MK1) vs `1EA907567*` (MK2 — different radar gen)

### 5.9 MK1 → MK2 (`MEB_GEN2`) deltas

| Item | MK1 (2021-23) | MK2 (2024+) |
|---|---|---|
| chassis code | `E2` | `E8` |
| op10 platform | `VOLKSWAGEN_ID4_MK1` | `VOLKSWAGEN_ID4_MK2` |
| DBC file | `vw_meb` | `vw_meb_2024` |
| `ESC_51` length | 48 B | **64 B** (extra wheel-tick / ABS state) |
| `Motor_51` length | 32 B | **48 B** (extra battery / regen telemetry) |
| `MEB_Camera_09` | 8 B | 3 B stub (replaced by other channels) |
| `QFK_01` CRC | shared LUT | **alt_crc_variant_1** (own LUT). Same 8H2F base, different magic-byte tables on QFK_01 / ESC_51 / Motor_51 |
| Side-assist bus | extended (cam_cp) | **pt_cp** (gateway integrated) |
| Camera HW | MFC4 (Continental) | MFC5 (Continental) — adds DSSS ahead-of-curve |
| Side-radar HW | rear bumper, 24 GHz | rear bumper, 77 GHz — better motorbike separation |
| Travel Assist | TA 2.5 | TA 3.0 (mandatory cap-wheel via `KLR_01`) |
| Battery telemetry | EM1_01 (single voltage report) | EM1_HYB_13 + MEB_HVEM_02 (split high/low) |
| Gear shifter source | `Gateway_73.GE_Fahrstufe` | same (`ALT_GEAR` flag pivots to `Getriebe_11`) |

### 5.10 Facelift DBC changes (`vw_meb_2024.dbc`)

Most impactful renames/additions, ranked:

1. `MEB_ACC_01` (768) → `ACC_19` — primary ACC HUD/control rename
2. `MEB_Distance_01` (591) → `Strukturen_01` — radar object list rename
   (would break our `radar_interface.py` on facelift)
3. New `MEB_AWV_01` (380196013) — facelift AEB HUD
4. New `MEB_PACC_01` (401604695) — predictive ACC
5. Whole new block of extended-ID/CAN-FD frames: `VZE_04`, `SAL_01`,
   `MEB_Camera_05..14`, `IPA_02`, `MEB_Radar_Unknown_01`, `EML_02`,
   `Charging_*`, `Battery_*`
6. `ESC_51` 48→64 B, `Motor_51` 32→48 B (more signals packed)

### 5.11 ID.4 MK1 spec divergence — none numerically vs op10

| Field | Ours | Op10 |
|---|---|---|
| mass / wheelbase / sR / centerToFrontRatio | identical | identical |
| chassis_codes | `{E2}` | `{E2}` |
| WMIs | `{EUROPE_SUV}` | `{USA_SUV, EUROPE_CAR, EUROPE_SUV}` ← **3 missing** |
| CarDocs | ID.4 only | ID.4 + ID.5 (same E2 chassis) |

US-built ID.4s and many European-VIN ID.4s currently fail to match because
`USA_SUV` and `EUROPE_CAR` WMIs aren't on our platform.

### 5.12 Coverage gaps for MK2 in this port

| Gap | Severity | Action |
|---|---|---|
| No `vw_meb_2024.dbc` | blocks MK2 | port from op10 |
| No `VolkswagenFlags.MEB_GEN2 (=128)` | blocks MK2 | add flag |
| No `VOLKSWAGEN_ID4_MK2` platform config | blocks MK2 | add with chassis E8 + `MEB_GEN2` |
| Length-branched safety RX checks missing | MK2 RX checks fail | add `volkswagen_meb_gen2_rx_checks` (ESC_51=64, Motor_51=48) |
| No `alt_crc_variant_1` CRC LUTs in safety | MK2 CRC fails | port LUT branches. `volkswagen_meb_compute_crc` already dispatches per addr; add a flag to swap the magic-byte tables on QFK_01/ESC_51/Motor_51 |
| No `VOLKSWAGEN_ID5_MK1` (E2) | minor | bundle under ID4_MK1 docs |
| No `VOLKSWAGEN_ID3_MK1/MK2` | medium | not in scope |
| `enableTrafficSignAssist`, `STOCK_KLR_PRESENT`, MEB_Camera_* fingerprint differ | MK2-specific | gate via GEN2 flag |

Lateral and longitudinal logic itself does NOT need to change for MK2.
Same EPS, same accel envelope.

### 5.13 Other VW MEB flags op10 has, we don't

- **MEB_GEN2** — facelift indicator
- **MQB_EVO** — Golf MK8/Leon MK4. Treated alongside MEB everywhere
  (`MEB | MQB_EVO`)
- **ALT_GEAR** — gear shifter source: `Gateway_73.GE_Fahrstufe` if set,
  else `Getriebe_11.GE_Fahrstufe`. Detected via fingerprint
- **DISABLE_RADAR** — for openpilot-long without stock radar; skips
  `AWV_03.FCW_Active`, forces `acc_type=2`, gates `alphaLongitudinalAvailable`

### 5.14 Order-of-work recommendation

| # | Task | Effort | Risk | Why |
|---|------|--------|------|-----|
| 1 | Add `USA_SUV` + `EUROPE_CAR` WMIs to ID.4 MK1 | 1 line | none | unblocks US/EU customers whose VIN doesn't match |
| 2 | Add `"E8"` to `chassis_codes` on `VOLKSWAGEN_ID4_MK1` | 1 line | none | unblocks US-built ID.4 |
| 3 | Add up/down to `create_acc_buttons_control` | ~5 lines | low | usable speed-set without enabling op long |
| 4 | Add `VOLKSWAGEN_ID3_MK1` (E1, 1935 kg) | ~10 lines | low | same MEB stack, smallest second platform |
| 5 | Add `SKODA_ENYAQ_MK1` + `AUDI_Q4_MK1` | ~15 lines each | low | piggybacks on (4) |
| 6 | Read `KBI_MFA_v_Einheit_02` for unit-aware cluster offset | small | medium | needs bench data |
| 7 | RE-log MEB_Side_Assist_02 + RCTA_01 | days | none | prereq for side tracks |
| 8 | Port `MEB_GEN2` + `vw_meb_2024.dbc` for ID.4 MK2 | ~150 lines | medium | needs a MK2 to test |

**Skip indefinitely:** `MEB_PACC_01`, `speed_limit_manager.py`, sunnypilot
ICBM.


---

## 6. Travel Assist / Driver Monitoring escalation chain

VW's "Emergency Assist" (Notbremsassistent für unaufmerksame Fahrer / EA)
is a multi-phase fallback that activates when Travel Assist detects no
driver hands for too long. The whole chain runs in the **MEB camera/EA
controller**, publishes on **bus 2**, and is gateway-forwarded to bus 0.
This is the critical section: replicating VW's stock attention-cascade
behavior in openpilot DM. The progression is split across **two
FAS-controlled messages** plus KLR (capacitive wheel), LDW (camera HUD),
AWV (AEB), PreCrash (collision actuation), and Blinkmodi (hazards). Most
signals are **fully documented** in the DBC; a handful are RE-only.

### 6.1 The three layers

- **MFC** (front camera, R242). Owns Travel Assist, Lane Assist, ACC
  request generation, FCW. Publishes `TA_01`, `MEB_ACC_01`, `MEB_Camera_*`,
  `LDW_02`, `EA_01`, `EA_02`.
- **EA controller** (Emergency Assist, often co-located with MFC on MEB).
  Drives the escalation phases. Same ECU owns the Notbremsassistent (AEB)
  outputs on `AWV_03` and indirectly `Airbag_01.AB_MKB_*`.
- **KLR** (Kapazitives Lenkradtouch-Steuergerät). Capacitive hands-on
  detection. Only present on certain trims (Travel Assist 3.0). MK1 base
  trims do not have it; later MY24+ ID.4 + GTX usually do.

### 6.2 Hands-on detection inputs

| Source | Signal | Notes |
|---|---|---|
| `LH_EPS_03.EPS_Lenkmoment` | bit 40, 10-bit signed | Driver torque, the universal "hands-on" signal. We already use for `steeringPressed`. Threshold ~0.3 Nm sustained. |
| `QFK_01.LatCon_HCA_Status` | EPS state machine | When hands-off persists, transitions through `PREEMPTED → REJECTED` (now treated as transient) |
| `KLR_01.KLR_Touchintensitaet_{1,2,3}` | bits 16/24/32, 0–250 | Facelift capacitive grip, three rim segments. Hands-on threshold ~30/250 per zone. Drivers using AliExpress capacitive-defeat dongles set the value to a constant >0 to fool the cluster |
| `KLR_01.KLR_Touchauswertung` | bits 40–43, 0–15 | Aggregated decision/score. **No VAL_ enums published**. On route 000000d8: only 0 (no touch, ~60% frames) and 7 (light touch, ~40% frames) observed. Stock route ground truth: cycles between 0, 7 (light touch), 10 (firm touch), boot-up 14 |
| `KLR_01.KLR_Lokalaktiv` | bit 14 | Local detector active |
| `KLR_01.KLR_Fehler/_ResponseError/_Fehler_Codierung` | 1-bit each | Sensor faults |
| `EA_02.ACF_Lampe_Hands_Off` | bits 16–17 | Cluster lamp request: `0=off, 1=Hands_Off_erkannt`. **Empirical**: never fired in f3/f4 routes — cluster handles its own hands-off icon internally; this signal may only be used by retrofit displays |

VW combines steering torque AND capacitive sensor; **either** above
hysteresis resets the timer.

#### KLR spoof (op10, gated by `STOCK_KLR_PRESENT`)

`mebcan.create_capacitive_wheel_touch(packer, bus, lat_active,
klr_stock_values)` rewrites:
```
KLR_Touchintensitaet_1 = 80
KLR_Touchintensitaet_2 = 200
KLR_Touchintensitaet_3 = 10
KLR_Touchauswertung    = 10  (matches stock "firm touch")
```
when `lat_active=True`. Sent on **both `pt` and `cam`** so gateway and
downstream listeners both see the spoof. Sent when stock COUNTER changes
(carcontroller.py:163-171 in op10). Validated against stock ground truth:
op10's value 10 matches "firm touch" class.

KLR_01 fails CRC with `vw_meb` checksum — message uses a **VW MQB-style
CRC variant** that opendbc's stock CRC doesn't compute. Read-only is fine,
but **TX** needs a per-message CRC override (same MQB used for HCA_01
forwarding). Sunnypilot does this for the spoof.

### 6.3 EA escalation phases — `EA_01.EA_Funktionsstatus` (msg 0x1A4 = 420, 8 B, bus 2, ~10 Hz)

| Value | Phase | Theoretical meaning |
|---|---|---|
| 0 | `EA_INIT` | boot/init |
| 1 | `EA_OFF` | system off (no engagement, e.g. cruise off) |
| 2 | `EA_STANDBY` | armed, monitoring (this is what 100 % of routes show) |
| 3 | `EA_PHASE0_AKTIV` | first warning ("Hands on wheel" prompt) ~10–15 s — **empirically skipped**; silent HUD prompt comes from `LDW_02.LDW_Texte=8` while EA stays in STANDBY |
| 4 | `EA_PHASE1_AKTIV` | second warning + audible chime ~20 s — **empirically skipped**; chime comes from `LDW_02.LDW_Gong=1` while EA stays in STANDBY |
| 5 | `EA_PHASE2_AKTIV` | belt jerk + brake handoff. Empirically the first EA phase actually entered: `EA_Gurtstraffer_Anf=Haptik_3` for ~750-1000 ms, `EA_Texte=2 "Nothalt aktiv übernehmen"`, `EA_Infotainment_Anf=2`, and `TSK_Status` flickers to `brake_only` while `ACC_18` delivers a ~−3.7 m/s² wake brake. Beep on `LDW_Gong=2`. `EA_Warnruckprofil` stays 0 |
| 6 | `EA_PHASE3_AKTIV` | adds hazards + active deceleration. `EA_Texte=3 "Nothalt durchgeführt wird"`, `EA_Blinken=3`, `Blinkmodi_02.BM_Warnblinken=1`. If car reaches standstill (not captured) should escalate to `EA_Anforderung_HMS=halten/parken`, doors unlock, autonomous honk via `TM_01.TM_Nur_Hupen` |
| 7 | `EA_REVERSIBLER_FEHLER` | recoverable EA fault, comes back after re-arm |
| 8 | `EA_IRREVERSIBLER_FEHLER` | latched fault, restart car |

Phase transitions are NOT controlled from openpilot. The MFC/EA ECU drives
them based on its own hands-on confidence (camera + KLR + steering torque).
Time thresholds are coded into the camera, vary by speed and trim, but
state-machine progression is invariant.

**Empirical progression (this car/firmware): `STANDBY → PHASE2 → PHASE3`.
PHASE0/PHASE1 reserved enums but unused.** "First warning" stages delivered
via `LDW_02` while EA stays in STANDBY.

### 6.4 EA action signals — `EA_01` (msg 0x1A4, bus 2, also bus 0 + bus 1 simultaneously)

| Signal | Bits | Width | Values |
|---|---|---|---|
| `CHECKSUM` / `COUNTER` | 0 / 8 | 8 / 4 | CRC + counter (signed by gateway when stock) |
| `EA_Parken_beibehalten_HMS` | 12 | 2 b | `0 nicht_beibehalten, 1 beibehalten, 2 init, 3 fault` — instructs EPB whether to hold after stop |
| `EA_Warnruckprofil` | 28 | 3 b | `0 keine, 1 Profil_1 … 7 Profil_7` — wake-up brake jerk profile dispatched into ESP. **Empirically stayed 0 in both DM-progression routes** — brake jolt is delivered via `ACC_18 + TSK_Status=brake_only`, not via this signal |
| `EA_eCall_Anf` | 31 | 2 b | `0 keine, 1 Ausloesen` — auto-dial emergency services |
| `EA_Funktionsstatus` | 40 | 4 b | the state machine above |
| `EA_Gurtstraffer_Anf` | 44 | 2 b | `0 keine, 1 Haptik_1, 2 Haptik_2, 3 Haptik_3` — reversible seatbelt pretensioner ("seatbelt jerk"). **Empirically the very first jerk on PHASE2 is already Haptik_3 (max)** — no graduated 1→2→3 progression on this software level |
| `EA_Anforderung_HMS` | 48 | 3 b | `0 none, 1 halten, 2 parken, 3 halten_Standby, 4 anfahren, 5 Loesen_ueber_Rampe, 6 Parken_mit_P` |
| `EA_Sollbeschleunigung` | 53 | 11 b | scale 0.005, offset −7.22 → range −7.22..+3.005 m/s². Sentinel `2046` = neutral, `2047` = error. EA-commanded accel during Phase2/3. Empirically only goes to **−2.0 m/s²** even at PHASE3 — actual wake-up brake delivered via `ACC_18` instead |

### 6.5 EA HUD/cosmetic — `EA_02` (msg 0x1F0 = 496, bus 2, ~10 Hz)

| Signal | Bits | Width | Values |
|---|---|---|---|
| `EA_Texte` | 12 | 4 b | Cluster banner — see complete enum below |
| `ACF_Lampe_Hands_Off` | 16 | 2 b | LED color: off / yellow / red. **Empirical: never seen** (cluster handles internally) |
| `EA_Infotainment_Anf` | 22 | 2 b | `0 init, 1 keine_Absenkung, 2 Absenkung (mute), 3 mute`. Fires same instant as `EA_Texte=2` |
| `EA_Tueren_Anf` | 24 | 1 b | unlock doors after Nothalt |
| `EA_Innenraumlicht_Anf` | 25 | 1 b | cabin light on after Nothalt |
| `zFAS_Warnblinken` | 26 | 2 b | `0 Aus, 1 Statisch, 2 Taster, 3 Statisch_ohne_WBT` — hazard request from EA. **Empirical: stayed 0 even in PHASE3** — `EA_Blinken=3` is the actual hazards trigger |
| `STP_Primaeranz` | 28 | 3 b | EA primary display: `0 keine, 1 Verfuegbar, 2 Aktiv, 3 Uebernahme, 4 Aktiv_Warnung, 5 Nicht_Verfuegbar` |
| `EA_Bremslichtblinken` | 31 | 1 b | brake-light flashing (during emergency brake). **Empirical: stayed 0** |
| `EA_Blinken` | 32 | 3 b | `0 kein_Blinken, 1 Wechselblinken_links, 2 Wechselblinken_rechts, 3 Warnblinken, 4 Warnblinken_Taster`. **Empirical: fires only at PHASE3 entry** |
| `EA_Unknown` | 60 | 3 b | op10 sets to `1` when hiding error, `3` when in error |

#### `EA_02.EA_Texte` complete enum

| Code | Text |
|---|---|
| 0 | `keine_Anzeige` |
| 1 | `Nothalteassistent_fehlende_Fahreraktivitaet` (PHASE 0/1 nag) |
| 2 | `Nothalteassistent_aktiv_Fahrzeugfuehrung_uebernehmen` (PHASE 2) |
| 3 | `Nothalteassistent_automatischer_Nothalt_wird_durchgefuehrt` (PHASE 3 entry) |
| 4 | `Nothalteassistent_automatischer_Nothalt_durchgefuehrt` (PHASE 3 exit, stopped) |
| 5 | `Nothalteassistent_Verbindung_zum_Notruf_wird_aufgebaut` (eCall placing) |
| 6 | `Nothalteassistent_deaktiviert` |
| 7 | `Nothalteassistent_Eingriff_abgebrochen` |
| 8 | `fehlende_Fahreraktivitaet_2` |
| 9 | `Sekundenschlaf_erkannt` ← **microsleep!** |
| 10 | `LaneAssist_Lenkung_uebernehmen` ← LKAS takeover prompt openpilot wants |
| 11 | `ACA_Fahrzeugfuehrung_uebernehmen` |
| 12 | `EA_Fahr_Standstreifenwechsel` |
| 14 | `nicht_verfuegbar_reversibel` |
| 15 | `Stoerung_irreversibel` |

**Empirical**: only codes **2 and 3** observed in DM/SOS escalation. Code 1
("fehlende Fahreraktivität") was not seen, suggesting cluster shows static
hands-off icon during PHASE 0/1 and only puts the banner up at PHASE 2.
Code 10 ("LaneAssist Lenkung uebernehmen") fires only on lateral-only nag
(e.g. lane-keep can't follow), not in DM/SOS.


### 6.6 Lateral steering authority during EA — `HCA_01` (msg 0x126 / 294, 8 B, bus 2, 1 Hz)

Sent BY the camera, relayed via panda. Can be intercepted/modified (relayed
in our safety TX list).

| Signal | Bits | Use |
|---|---|---|
| `EA_ACC_Sollstatus` | 25 \| 2 | EA tells ACC: `0=normal, 1=hold-after-stop, 2=takeover-prep, 3=takeover-active` |
| `EA_Ruckprofil` | 27 \| 3 | jerk profile output (same enum as `EA_Warnruckprofil`) |
| `EA_Ruckfreigabe` | 40 \| 1 | EA grants jerk release to the EPS. **Empirical: stayed 0 throughout** |
| `EA_ACC_Wunschgeschwindigkeit` | 41 \| 10 (0.32 km/h) | EA's target speed (drops to 0 during pullout) |
| `HCA_01_LM_Offset` | 16 \| 9 (0.01 Nm) | torque request from EA into the EPS (during shoulder maneuver) |
| `HCA_01_LM_OffSign` | 31 \| 1 | sign |
| `HCA_01_Vib_Freq` / `HCA_01_Vib_Amp` | 12 \| 4 / 36 \| 4 | wheel vibration (additional haptic warning), 1 Hz / 0.2 Nm |
| `HCA_01_Enable` | — | Camera-side HCA enable |
| `HCA_01_Standby/Request/Available` | — | HCA state machine |

### 6.7 Travel Assist orchestrator — `TA_01` (msg 0x26B / 619, 8 B, bus 2)

| Signal | Bits | Values | Meaning |
|---|---|---|---|
| `Travel_Assist_Status` | 13 \| 3 | `0=disabled, 2=ready, 3=pre_ready, 4=enabled` | TA state |
| `Travel_Assist_Request` | 19 \| 3 | `0=no_request, 1=error, 3=disable, 4=enable` | Request to TA |
| `Travel_Assist_Available` | 23 \| 1 | `0/1` | Gating |

Op10 lists `MSG_TA_01 (0x26B)` in `VW_MEB_LONG_TX_MSGS`; carport may TX
this when openpilot owns lateral. Role: tells the cluster "this lane-keep
is acting like Travel Assist" so green-lane HMI shows correctly.

`Travel_Assist_Status` empirically cycles between **2 (ready), 3
(pre_ready), 4 (enabled), 0 (disabled)**. Brief `Stat=3 (pre_ready)` (~0.5
s) appears during clean disengage transitions before settling to `Stat=2`.
**Falls back to 2 (ready) at PHASE2 entry** — this is how the system marks
itself unavailable.

### 6.8 PEA — Predictive Energy Assistant takeover — `Motor_41` (msg 0x2C2 = 706, 8 B, bus 1, 2 Hz)

| Signal | Use |
|---|---|
| `PEA_Texte` | `0=none, 1=PEA_Fahreruebernahme_noetig` (drivetrain asks driver to take over), `2=Reku_nicht_verfuegbar` (regen unavailable) |

Electric powertrain-side take-over request (e.g. on regen failure or
thermal limit). Independent of EA.

### 6.9 KLR_01 detail — `KLR_01` (0x25D / 605, 8 B, bus 0, 16.7–50 Hz)

(Cross-ref §6.2.) Per-zone capacitive intensity 0..250. On route 000000d8
(highway, both hands sometimes off the wheel) max intensity per zone was
36/33/41 — driver touched lightly only. Touchauswertung=7 for ~40% of frames.

In facelift routes, max touch value of 41/250 (light grip). Key signal for
facelift hands-on detection.

### 6.10 Camera HUD channel — `LDW_02` (BO 919, 10 Hz, `vw_meb.dbc:840-860`)

| Signal | Bits | Semantics |
|---|---|---|
| `LDW_Texte` | 16 \| 4 | Cluster takeover text. DBC documents only `0`, `4 laneAssistTakeOverUrgent`, `8 laneAssistTakeOver`. Op10's MQB enum suggests `1 Unavail+Chime`, `3 Unavail_NoSensor+Chime`, `6 emergencyAssistUrgent`, `7 laneAssistTakeOver+Chime`, `9 emergencyAssistChangingLanes`, `10 deactivated` — **likely valid on MEB but unconfirmed** |
| `LDW_Gong` | 12 \| 2 | `0 None, 1 Chime, 2 Beep` (VAL 2478). Audible chime channel |
| `LDW_Vib_Amp_VLR` / `_Anlaufzeit_VLR` / `_Anlaufsp_VLR` | 28 \| 4 / 32 \| 4 / 24 \| 4 | **EPS rim haptic-vibrate** amplitude/duration/spool-up. Steering-rim shake. Available as a softer haptic substitute for the belt-pulse (not used by stock cascade) |
| `LDW_Status_LED_gelb / _gruen` | 61 / 62 | Yellow/green wheel-icon LED |

### 6.11 Empirical EA escalation traces — routes f3 + f4

These are the **canonical measured ground truth**. Two stock routes the
user captured: `f4` (61 s windows) and `f3` (61 s windows). Both routes
exhibit identical signal choreography. Driver intervened within ≤2.5 s of
PHASE3 in every case, so **no PHASE3 hold/HMS/eCall/horn signals fired** —
those need a "let it run to standstill" capture.

**Phase coverage**

| Route | PHASE 2 (`EA_F=5`) frames | PHASE 3 (`EA_F=6`) frames | full SOS stop reached? |
|---|---:|---:|---|
| f3 | 756 | 60 | no — driver always recovered |
| f4 | 718 | 252 | no — driver always recovered |

#### Decoded timeline — route `000000f4--70de8333d7` (PHASE 2 + PHASE 3 with takeover before SOS)

| t (s) | event | signals that flipped |
|------:|-------|----------------------|
| 0.38 | system arms | `EA_Funktionsstatus 0→2 STANDBY`, `EA_Sollbeschleunigung 0→3.01` (= idle/Neutralwert) |
| 15.92 | **silent prompt** | `LDW_Texte 0→8 laneAssistTakeOver` (silent icon) |
| 20.92 | **red + chime (5 s after silent)** | `LDW_Texte 8→4 laneAssistTakeOverUrgent` (red), `LDW_Gong 0→1 Chime`, `EA_Sollbeschleunigung 3.01→0` |
| **30.87** | **active intervention (10 s after chime)** | `EA_Texte 0→2 Fahrzeugfuehrung_uebernehmen`, `EA_Infotainment_Anf 0→2 Absenkung` (radio mute), `EA_Funktionsstatus 2→5 PHASE2_AKTIV`, `EA_Gurtstraffer_Anf 0→3 Haptik_3` (max belt pulse, no 1→2→3 ramp), `EA_Sollbeschleunigung 0→-1.0` m/s² (wake-up brake), `LDW_Texte 4→0`, `LDW_Gong 1→2 Beep` |
| 31.90 | belt pulse ends | `EA_Gurtstraffer_Anf 3→0` (~1 s pulse) |
| 33.04 | **driver took over** | `EA_Infotainment_Anf 2→0`, `EA_Funktionsstatus 5→2 STANDBY`, `EA_Sollbeschleunigung -1→3.01` |
| 47.12 | **second event begins** | `LDW_Texte 0→8` |
| 52.12 | second chime (also 5 s later) | `LDW_Texte 8→4`, `LDW_Gong 0→1` |

Route `000000f3--69d35c80a9` (61 s): monitoring baseline + same shape.
`Funktionsstatus 0→2` at 0.34 s, then steady state. `AWV_03.FCW_Active`
fired 262 frames (one or two FCW pings). No EA events (in that 61 s
window).

#### Per-event timeline (route f3, event 2 @ t=329.18s — typical full PHASE 2 → PHASE 3)

```
t = 329.18s   EA_Texte = 0→2 ("Fahrzeugführung übernehmen")
              EA_Infotainment_Anf = 0→2 (audio fade)
t = 329.19s   EA_Funktionsstatus = 2→5 (PHASE 2)
              EA_Gurtstraffer_Anf = 0→3 (Haptik_3 — seatbelt yank)
              EA_Sollbeschleunigung = 3.01 → −1.83 m/s²  (immediate)
t = 329.21s   ACC_Akustischer_Fahrerhinweis = 0→3 (audible chime)
              ACC_Optischer_Fahrerhinweis = 0→1 (visual indicator)
t = 329.29s   Travel_Assist_Status = 4→2 (TA disengaged)
t = 330.21s   EA_Gurtstraffer_Anf = 3→0 (yank ends — duration 1.02 s)
t = 330.23s   ACC_Akustischer_Fahrerhinweis = 3→0 (chime ends — duration 1.02 s)
t = 331.25s   ACC_Optischer_Fahrerhinweis = 1→0
t = 331.5s    *** WAKE-UP BRAKE: ESC Brake_Pressure spikes to 0.78 bar for ~0.4 s ***
              EA_Sollbeschleunigung is at −1.78 m/s²; ESP responds with brief active brake
t = 333.6s    EA_Sollbeschleunigung gradually ramps to −1.0 m/s² (relaxing)
t = 334.19s   EA_Texte = 2→3 ("automatischer Nothalt wird durchgeführt")
              5.01 s after PHASE 2 entry
t = 334.21s   EA_Funktionsstatus = 5→6 (PHASE 3)
              EA_Sollbeschleunigung = −1.0 → −2.00 m/s² (committed deceleration)
t = 334.23s   EA_Blinken = 0→3 (Warnblinken — hazard lights on)
t = 334.81s   driver intervened
              EA_Funktionsstatus = 6→2, EA_Sollbeschleunigung = 3.01 (back to neutral)
t = 334.83s   EA_Blinken = 3→0
t = 335.61s   EA_Texte = 3→0
```

#### Sub-trace: PHASE2 wake-up sequence shared across both routes (≈ identical shape)

| t (rel) | event | signals |
|---|---|---|
| **t = 0** | first warning text appears | `EA_Texte = 2 (Fahrzeugführung übernehmen)`, `EA_Infotainment_Anf = 2` |
| t + 0.01 s | PHASE2 entry (one frame after text) | `EA_Funktionsstatus = 5` |
| t + 0.01 s | **seatbelt jerk fires** | `EA_Gurtstraffer_Anf = 3` (haptik_3 — strongest tier observed) |
| t + 0.01 s | EA brake request begins | `EA_Sollbeschleunigung` jumps from `+3.01` (inactive) to **−1.0 to −1.67 m/s²** |
| t + 0.01 … 1.02 s | sustained seatbelt jerk + accel ramp | `EA_Gurtstraffer_Anf = 3` for ~1.0 s; accel ramps slowly more negative (−1.67 → −1.88 over the same window) |
| **t + 1.02 s** | seatbelt jerk releases | `EA_Gurtstraffer_Anf = 0` (sharp transition) |
| t + 2.0–2.1 s | **front-radar FCW + brake spike** | `VMM_02.FCW_Active = 1` (5–8 frames), `ESC_51.AEB_Breaking_02` jumps **126 → 156** |
| t + 2.05 s | hard brake actuation begins | `Brake_Pressure` ramps **0 → 32.6 %** in ~80 ms |
| t + 2.10–2.13 s | peak **AEB_Breaking_01 spike** | `AEB_Breaking_01` 0 → 213 (1 frame) → 41 → 0 (≈40 ms total — the "wake-up" jerk pulse) |
| t + 2.20 s | brake fades | `Brake_Pressure` ramps back toward 0 over 200 ms |
| t + 2.30 … 5.0 s | sustained mild deceleration | `EA_Sollbeschleunigung` lingers around −1.8, gradually ramps back toward −1.0 |

#### Sub-trace: PHASE2 → PHASE3 transition (Nothalt entry)

Triggered if PHASE2 expires without driver re-engagement (~5 s after PHASE2
entry on both routes). Deterministic timing.

| t (rel from PHASE2 start) | event | signals |
|---|---|---|
| ≈ +5.00 s | text changes | `EA_Texte`: `2` → `3` (Nothalt_durchgeführt_wird) |
| ≈ +5.02 s | **PHASE3 entry** | `EA_Funktionsstatus = 6` |
| ≈ +5.02 s | full brake authority | `EA_Sollbeschleunigung = -2.00 m/s²` (saturates at most negative seen) |
| ≈ +5.04 s | brake-light blinking begins | **`EA_Bremslichtblinken = 3`** (rapid pattern). NOTE meb6 says it stayed 0; meb2 says it fires → likely capture-window dependent |
| (during PHASE3) | NOT observed (driver intervened) | `EA_Anforderung_HMS = 0`, `EA_eCall_Anf = 0`, `MFL_Signalhorn = 0`, `SMLS_Hupe = 0`, `zFAS_Warnblinken = 0`, `ACF_Lampe_Hands_Off = 0`, `EA_Tueren_Anf = 0`, `EA_Innenraumlicht_Anf = 0` |
| PHASE3 exit | driver took over | `EA_Funktionsstatus`: 6 → 2 (STANDBY direct, no transition state). `EA_Bremslichtblinken` returns 0 within 1 frame. `EA_Texte = 3` lingers ~0.8 s after exit then clears |

In `f4`, after PHASE3 exit at 69.60 s there's a **secondary brake actuation**
at 69.63–70.05 s (`AEB_Breaking_01` spikes 52→245→252 then returns to 0 over
~400 ms with `Brake_Pressure` ≤ 1 %). Appears to be the driver's own brake
re-pressing on top of EA's release; not part of the EA chain.

### 6.12 Empirical cadence summary

```
t=0       silent    LDW_Texte=8                    Funktionsstatus=2
t+5s      chime     LDW_Texte=4 + LDW_Gong=1       Funktionsstatus=2
t+15s     ACTIVE    EA_Texte=2, EA_Funktionsstatus=5, LDW_Gong=2 Beep
                    EA_Gurtstraffer_Anf=3 (1s pulse)
                    EA_Sollbeschleunigung=-1.0 m/s² (continuous decel)
                    EA_Infotainment_Anf=2 (radio mute)
                    [if not taken over → eventually PHASE3 SOS:
                        Blinkmodi_02.BM_*, EA_Blinken=3, zFAS_Warnblinken=1,
                        EA_Tueren_Anf=1, EA_eCall_Anf=1, TM_Nur_Hupen=1]
```

### 6.13 The "fence" state (meb3 finding)

The first indication of a hands-off countdown is **`EA_Sollbeschleunigung`
dropping off Neutral (raw 2046) to ~1444 (−0.444 m/s²)** while
`EA_Funktionsstatus` is still 2 (STANDBY). Pre-arm appears **~10 seconds
before** PHASE2 entry. So the *real* timeline is:

| t− | Funktionsstatus | EA_Texte | Sollb | Gurtstraffer | Sound | Cluster |
|---|---|---|---|---|---|---|
| t−10 s | 2 STANDBY | 0 | **−0.444** (fence) | 0 | none | green wheel + nag |
| t=0 | **5 PHASE2** | **2 aktiv_uebernehmen** | −1.0 → −1.7 ramp | **3 Haptik_3** (1.0 s) | gong | red wheel + flashing |
| t+5 s | **6 PHASE3** | **3 Nothalt_durchfuehren** | **−2.0** | 0 | continuous | "Nothalt wird durchgeführt" |
| t+5 s+ε (driver intervenes here in observed routes) | 2 STANDBY | 0 | Neutral | 0 | clears | back to green |

Concrete observation, route f4: STANDBY-fence at t=20.86s → PHASE2 at
t=30.86s (+10.0 s) → PHASE3 at t=67.06s. Route f3: PHASE2 at 251.96s →
STANDBY → PHASE2 at 329.18s → PHASE3 at 334.20s (5.02 s gap PHASE2→PHASE3,
consistent across cycles).

### 6.14 Empirical corrections vs theoretical map

| Theoretical claim | Empirical observation |
|---|---|
| PHASE 0/1 nag exists | **NOT seen** — system jumps directly STANDBY (2) → PHASE 2 (5). Either cluster does PHASE 0/1 internally without changing `EA_Funktionsstatus`, or recordings begin past those phases |
| `EA_Warnruckprofil` carries the wake-up brake | **WRONG** — stayed 0 throughout. Wake-up brake is delivered via `EA_Sollbeschleunigung` (commanded to −1.83 m/s² immediately at PHASE 2 entry) and the brake controller responds with a ~0.8-bar / 0.4-s ESP pulse. Likewise `HCA_01.EA_Ruckfreigabe` stayed 0 |
| `EA_Gurtstraffer_Anf` carries the seatbelt jerk | **CONFIRMED** — `Haptik_3` (value 3) fires for **1.02 s** at PHASE 2 entry, exactly once per event |
| `EA_Gurtstraffer_Anf` ramps 1→2→3 | **WRONG** — straight to maximum (Haptik_3) |
| Audible chime via `ACC_Akustischer_Fahrerhinweis = 1` | **WRONG** — value **3** (not the DBC-listed `0/1`!) for 1.02 s, perfectly synchronized with seatbelt yank. Update DBC range to `[0\|3]` and treat 3 as "warning chime, escalation level" |
| Visual indicator via `ACC_Optischer_Fahrerhinweis = 1` | **CONFIRMED** — fires alongside chime, slightly longer (~2 s) |
| Hazards via `EA_Blinken = 3` (Warnblinken) | **CONFIRMED** — fires only at PHASE 3 entry, not PHASE 2 |
| `zFAS_Warnblinken` for hazards | **WRONG** — stayed 0; `EA_Blinken = 3` is the actual hazards trigger |
| `ACF_Lampe_Hands_Off` for hands-off cluster lamp | **NEVER SEEN** — cluster handles its own hands-off icon internally |
| PHASE 3 deceleration starts at −3 m/s² | **WRONG** — empirically −2.0 m/s² steady |
| `EA_Texte` codes 1 and 2 used | **only 2, 3 observed** — code 1 ("fehlende Fahreraktivität") was not seen, suggesting cluster shows static hands-off icon during PHASE 0/1 and only puts the banner up at PHASE 2 |
| `Travel_Assist_Status` stays = 4 during EA | **WRONG** — TA falls back to 2 (ready) at PHASE 2 entry |
| **Radio attenuation `EA_Infotainment_Anf = 2`** | confirmed part of PHASE2 entry — flagged here, missing from earlier stage tables |
| **No `LDW_Vib_Amp_VLR`** | no rim shake on this escalation. Steering haptic not used in stock cascade |
| **No `Blinkmodi_02.*`, no `EA_Blinken`, no `zFAS_Warnblinken` at PHASE2** | hazards do NOT come on at PHASE2. Wait for sustained PHASE2 or for PHASE3 |
| **`STP_Primaeranz`, `EA_Tueren_Anf`, `EA_eCall_Anf` all stayed 0** | only fire at PHASE3, which neither route reached |
| **`PEA_Texte` (Motor_41), `SC_PreCrash_Warnung` (PreCrash_02) all 0** | Predictive EA channel and PreCrash actuation never engaged. Reserved for collision-imminent, not driver inattention |
| **No honking observed** | `TM_Nur_Hupen` (BO 1447) and `MFL_Signalhorn` (BO 980) both stayed 0. **Auto-honk only fires at PHASE3 SOS standstill** — not captured |
| PHASE 2 vs PHASE 3 = "warning jerk vs emergency stop" | **WRONG** — actually "belt+brake handoff vs add-hazards-to-the-mix" |
| HUD chime delivered via EA | **WRONG** — `LDW_02.LDW_Gong` is the channel. EA has its own infotainment-mute and ACC has its own one-shot beep |

### 6.15 Other text values seen on route

| Field | Code | Frames | Meaning |
|---|---|---|---|
| `ACC_Texte_Zusatzanz_02` | 1 | 252 (f4) | "ACC_AUS" |
| `ACC_Texte_Zusatzanz_02` | 3 | 1986 (f4) / 2267 (f3) | "UEBERTRETEN" (overspeed) |
| `ACC_Texte_Zusatzanz_02` | 4 | 114 (f4) | "ABSTAND" (gap warning) |
| `Travel_Assist_Status` | 3 | 201 (f4) / 121 (f3) | `pre_ready` (transient) |
| `KLR_Touchintensitaet_*` | max ~44/250 | both routes | facelift capacitive grip — light touch present even on this car |

### 6.16 Honking analysis — `TM_01` BO 1447 (Telematics)

| Signal | Bits | Enum |
|---|---|---|
| `TM_Nur_Hupen` | 48 \| 1 | `0 inaktiv, 1 aktiv` (VAL line 3148) |
| `TM_Spiegel_Anklappen` | 47 \| 1 | mirror fold |

DBC search found three horn-related signals; only `TM_01.TM_Nur_Hupen` is
plausibly the auto-horn:

| BO_ | Name | Signal | Direction | Verdict |
|---|---|---|---|---|
| 817 | MFL_01 | `MFL_Signalhorn` | RX (driver button) | not auto-horn |
| 980 | SMLS_01 | `SMLS_Hupe` | RX (stalk button) | not auto-horn |
| **1447** | **TM_01** | **`TM_Nur_Hupen`** (bit 48, 1-bit, 0=inaktiv 1=aktiv) | **TX-able by Telematics module** | **prime candidate** |

Driven by the **telematics module** during SOS auto-honking after Nothalt.
Separate from `MFL_Signalhorn` (driver wheel-button, BO 980 b56) and
`SMLS_Hupe` (alarm-system horn chirp, BO 980 b29). For openpilot to drive
SOS-style auto-honk we'd need to inject `TM_01` — gateway/BCM acceptance
**untested**.

`TM_01.TM_Nur_Hupen` was 0 in both f3/f4 routes — SOS phase that arms it
never sustained long enough. Confirming this requires a route where
PHASE3 runs to vehicle standstill (driver does nothing for ~30 seconds
straight).

**Hypothesis on the honking trigger** (needs confirmation): the BCM (J519)
physically drives the horn relay. EA may signal the BCM via either:
1. A signal in `EA_02` not yet decoded (the message has 3 reserved bytes
   32–60 and `EA_Unknown` field at bit 60)
2. A direct UDS service call from EA controller to BCM (not visible on
   powertrain CAN)
3. `EA_02.EA_Blinken = 4` (`Warnblinken_Taster` — distinct from value 3)
   — never seen here, may be the post-stop hazard mode that ALSO triggers
   horn

To confirm: record a full-stop SOS event by letting EA escalate to PHASE 3
*and* coast to vEgo == 0. Once stopped, expect the horn to fire ~5–10 s
later if no driver input.


### 6.17 Bus topology of EA messages

- **`EA_01` and `TA_01` emit on bus 0, bus 1, AND bus 2 simultaneously**
  with identical content. Panda sees the same signal regardless of relay side.
- The signals OP would *spoof* (`KLR_01`, `EA_02`) op10 transmits to
  **both bus 0 and bus 2** for redundancy.

### 6.18 EA's brake is its own channel — not AEB

| Channel | State during DM escalation |
|---|---|
| `AWV_03.Pre_FCW` | always 0 |
| `AWV_03.FCW_Active` | always 0 |
| `MEB_AWV_01.Distance_Warning / FCW_Display / FCW_Sound` | always 0 |
| `VMM_02.AEB_Active` | always 0 |

The hard-brake during EA escalation is **NOT** routed through Front Assist.
Front Assist stays armed in parallel and would override if a forward
collision threat appears, but EA's emergency stop is its own braking
channel via `EA_01.EA_Sollbeschleunigung → TSK`.

**EA's accel channel must not collide with OP's ACC_18.** Both go to TSK.
When `EA_Funktionsstatus ∈ {3,4,5,6}` (any AKTIV phase), TSK uses
EA_01.EA_Sollbeschleunigung as the authoritative accel and ignores ACC_18.
Carstate should expose an `eaActive` flag so OP can defer to EA when
active rather than fight it.

### 6.19 Replication strategy — what we should send vs read

**Read-only (mirror in DM)**:

- From cam-CAN: `EA_01.EA_Funktionsstatus / EA_Warnruckprofil /
  EA_Gurtstraffer_Anf / EA_Sollbeschleunigung / EA_Anforderung_HMS`,
  `EA_02.EA_Texte / ACF_Lampe_Hands_Off / zFAS_Warnblinken /
  STP_Primaeranz`, `LDW_02.LDW_Texte / LDW_Gong`, `HCA_01.EA_Ruckfreigabe
  / EA_ACC_Sollstatus`, `AWV_03.FCW_Active / Pre_Brake_Fill`
- From PT-CAN: `KLR_01.KLR_Touchauswertung` + intensities,
  `Motor_41.PEA_Texte`, `Airbag_02.AB_Gurtschloss_FA` (already used),
  `Airbag_04.PreCrash_FAS_Fkt_Status / LGI_FAS_Fkt_Status /
  AbstWarn_MV_FAS_Fkt_Status`, `Blinkmodi_02.BM_Not_Bremsung /
  BM_Warnblinken / BM_Recas / BM_NBA_Status`, `PreCrash_02.SC_PreCrash_Warnung`
  (canonical FAS warning level 1–8) plus the actuation request bits.

**Send (drive cluster from our DM)**:

1. **Cluster text + chime** via `LDW_02` — already works
   (`LDW_Texte=8`/`4` plus `LDW_Gong=1`/`2`). Extend `LDW_MESSAGES` to
   include the inferred MEB codes `7 takeOver+Chime`, `6 emergencyAssistUrgent`,
   `9 emergencyAssistChangingLanes` if confirmed.
2. **Brake-jerk** via `MEB_ACC_01` accel pulse — already wired
   (`create_acc_accel_control`). 200 ms `actuators.accel = -2.5 m/s²`
   produces same Warnruck feel without spoofing `EA_01`.
3. **Steering-rim haptic** via `LDW_02.LDW_Vib_Amp_VLR /
   LDW_Anlaufzeit_VLR / LDW_Anlaufsp_VLR` — substitute for belt-jerk at
   stages 2–3. EPS executes directly. Available without spoofing FAS.
4. **Hazards** via `EA_02.EA_Blinken=3 Warnblinken` and
   `zFAS_Warnblinken=1`. Op10 already partially writes `EA_02` via
   `create_blinker_control` (`mebcan.py:56-83`). Needs gateway/BCM to
   accept our send. **Untested.**

**NOT replicable from CAN alone**:

- Reversible belt pretensioner pulse — `EA_Gurtstraffer_Anf` intent is
  observable, but the airbag SG executes via private body sub-CAN. Foreign
  `EA_01` sender will likely be rejected by SG arbitration. Substitute:
  steering-rim vibration (different sensation but available).
- Full AEB hard brake — ESC private/Flexray bus.
- Pyro pretensioner — one-shot, airbag-only.
- PreCrash actuation (window/sunroof/seatback/headrest/AFR) — gateway
  coding accepts only specific senders.
- eCall placement — separate emergency-call module, undefined behavior.
- Door-unlock after stop — comfort-CAN, gated.

#### Recommended scope for DM replication

Build a state machine in our DM that **mirrors `EA_Funktionsstatus`** and
drives:

- Stage 1–3: `LDW_Texte` + `LDW_Gong` cluster text/chime
- Stage 3 alt: `LDW_Vib_Amp_VLR` rim shake
- Stage 4: `MEB_ACC_01` accel pulse (-2.5 m/s² × 200 ms)
- Stage 5 substitute: stronger rim shake (no belt access)
- Stage 6: write `EA_02.EA_Blinken=3` + `zFAS_Warnblinken=1` (test if BCM honors)
- Stage 7: stop via `MEB_ACC_01.acc_hold_type = ACC_HMS_HOLD`, hold via
  ESC standstill — already in `mebcan.acc_hold_type`

Drop from scope: belt-jerk, doors, eCall, PreCrash actuation, full AEB.

#### What to implement per phase (op port action surface)

| Phase | OP action | Implementation surface |
|---|---|---|
| PHASE 0 visual | mirror to LDW HUD `LDW_Texte = laneAssistTakeOver` | already done ✓ |
| PHASE 1 audible | publish `MEB_ACC_01.ACC_Akustischer_Fahrerhinweis = 1` | new TX field, mebcan |
| PHASE 2 jerk | publish `EA_01.EA_Warnruckprofil = N`? | **probably not allowed** — EA_01 is a radar/camera output. Better: drive openpilot's own brake jerk via `ACC_18.ACC_Sollbeschleunigung_02` |
| PHASE 3 stop | publish ACC_18 with steady deceleration + LDW_Texte = takeover | already supported by long path |
| seatbelt jerk | publish `EA_01.EA_Gurtstraffer_Anf` | requires TX on EA_01 — probably outside scope |
| hazards | publish `EA_02.EA_Blinken = 3` | requires TX on EA_02 — outside scope |
| brake-light flash | publish `EA_02.EA_Bremslichtblinken = 1` | requires TX on EA_02 — outside scope |

**Key constraint**: openpilot can mirror the *user-facing* EA experience
via the messages we already TX (HCA_03, ACC_18, MEB_ACC_01, LDW_02). The
*physical* actuators (seatbelt pretensioner, hazards, brake-light flash,
eCall) are owned by the EA controller / SRS / BCM and not currently in
our TX whitelist. Adding them is a panda-safety change — needs
ALLOW_DEBUG and explicit safety review.

For a minimal first pass: **mirror visual + audible alerts via MEB_ACC_01
and LDW_02** (no panda change), and let stock AEB/EA continue to handle
physical actuation when openpilot is engaged. This is the design pattern
Tesla and Hyundai use for stock-AEB pass-through.

### 6.20 Quick reference — channels we send vs read for DM replication

| Channel | Send? | Read? | Notes |
|---|---|---|---|
| `LDW_Texte` (cluster text) | ✅ already wired | ✅ | values 4 (red), 8 (silent) confirmed; 6/7/9/10 inferred from MQB |
| `LDW_Gong` (chime/beep) | ⚠️ extend mebcan | ✅ | values 1 (chime), 2 (beep) confirmed |
| `LDW_Vib_Amp_VLR` (rim shake) | ⚠️ extend mebcan | – | not used by stock cascade — but available as a softer haptic |
| `MEB_ACC_01` accel pulse (wakeup brake) | ✅ via `actuators.accel` | – | `EA_Sollbeschleunigung = -1.0 m/s²` is the stock equivalent |
| `EA_Gurtstraffer_Anf` (belt jerk) | ❌ private body bus | ✅ | airbag SG executes; foreign sender likely rejected |
| `EA_02.EA_Blinken` (hazards) | ❌ untested | ✅ | gateway acceptance unknown |
| `TM_01.TM_Nur_Hupen` (auto-honk) | ❌ untested | ✅ | telematics-owned, gateway acceptance unknown |
| `EA_Tueren_Anf` (door unlock) | ❌ comfort-CAN | ✅ | not reachable |
| `EA_eCall_Anf` (eCall) | ❌ separate module | ✅ | undefined behavior — do not |
| `PreCrash_02.*` actuation | ❌ specific-sender | ✅ | windows/sunroof/seatback/headrest |

**In-scope for our DM replication:**
1. Cluster text + chime (LDW_02) — full
2. Wakeup brake via ACC accel pulse — full
3. Stronger rim haptic as belt-pulse substitute — bench-test only

**Out-of-scope (not reachable from openpilot CAN):** belt jerk execution,
door-unlock, eCall, PreCrash actuation, full AEB, SOS auto-honk — unless
we get bench-test evidence the gateway forwards our injection.

### 6.21 Implications for the openpilot port

1. **DM passive-read in carstate** should surface `EA_Funktionsstatus`,
   `EA_Sollbeschleunigung`, `EA_Texte`, `EA_Gurtstraffer_Anf`. State-of-the-art
   DM logic the cluster uses can be mirrored in OP's UI from these four
   signals alone.
2. **The "fence" state matters.** If openpilot wants to defeat / override
   the EA hands-off countdown (only useful when running OP lateral with
   `STOCK_KLR_PRESENT`), watching `EA_Sollbeschleunigung < Neutral` while
   `EA_Funktionsstatus == 2` gives ~10 s warning *before* PHASE2 fires
   Haptik_3. Cleanest hook for "tell openpilot to nag the driver before
   the car nags".
3. **EA's accel channel must not collide with OP's ACC_18.** See §6.18.
4. **For the spoof / suppression path** (only relevant with
   `STOCK_KLR_PRESENT` + DM hardware-defeat), op10's KLR spoof of
   `Touchauswertung=10` is correct per ground truth.
5. **The auto-horn replication** is straightforward once confirmed: TX
   `TM_01` with `TM_Nur_Hupen=1` from openpilot during emergency. Pending
   validation route.


---

## 7. AEB / Front Assist / PreCrash chain

The MEB AEB stack has **five independent layers** that escalate. Each is
a separate DBC message. Three independent ECUs participate, each
publishing its own AEB/FCW bit.

### 7.1 ECU/message map

| Msg | Bus | ECU | Active signals |
|---|---|---|---|
| `AWV_03` (0x0DB = 219, 48 B) | 2 + GW→0, 1 Hz | front radar (front-collision warner) | `FCW_Active` bit 64, `Pre_Brake_Fill` bit 76 |
| `MEB_AWV_01` (0x16A954AD = 380196013) | facelift only | camera | cluster HUD/sound mirror — `Distance_Warning, FCW_Display, AWV_Enabled, AWV_Init, FCW_Sound, Target` |
| `VMM_02` (0x139 = 313, 32 B, 50 Hz) | 0 + 1 | brake controller (VDM/MK100) | `AEB_Active` bit 31, `FCW_Active` bit 56, `Brake_Pressed_3` bit 48, `Brake_Pressure` bits 76–86 |
| `Parken_01` (0x206 = 518, 24 B, 50 Hz) | 2 | park-assist | `AEB_Active` bit 16 (low-speed parking AEB) |
| `ESC_51` (0x0FC = 252, 48 B, 100 Hz) | 0 + 1 + GW→2 | brake actuator (ESP) | `AEB_Breaking_01` (8 b), `AEB_Breaking_02` (8 b) — actual commanded brake pressure during AEB |
| `PreCrash_02` (0x13F = 319, 8 B, 5 Hz) | **only bus 1** | airbag SRS | full pre-crash actuator chain (below) |
| `Airbag_01` (0x40 = 64, 8 B, 50 Hz) | 0 | gateway | post-crash flags + MKB |
| `MEB_ACC_01` (0x300 = 768, 48 B) | 0 (TX'd by us) | — | cluster ACC alerts: `ACC_Optischer/Akustischer_Fahrerhinweis`, `ACC_Warnhinweis`, `ACC_Texte_Primaeranz_02`, `ACC_Texte_Zusatzanz_02`, `ACC_Wunschgeschw_Farbe`, `ACA_Querfuehrung` |
| `ESP_20` (0x659 = 1629) | — | ESC | `ESP_Stoppverbot_Anz_01 = "Notbremsung_aktiv"` indicates emergency-brake-active to other ECUs |

### 7.2 Stages — Euro NCAP / SSP nominal mapping (TTC values speed-dependent)

| Stage | Trigger | DBC signal | Expected behavior |
|---|---|---|---|
| 1 | Distance falling below comfort gap | `MEB_ACC_01.ACC_Optischer_Fahrerhinweis = 1` | Visual cluster icon ("Front Assist warning ⚠") |
| 2 | TTC ~3.6–2.6 s | `MEB_ACC_01.ACC_Akustischer_Fahrerhinweis = 1` + `ACC_Warnhinweis = 1` | Acoustic warning chime |
| 3 | TTC ~2.6 s, latent prewarning | `PreCrash_02.SC_PreCrash_Warnung = 1` ("latente_Vorwarnung") | Pre-crash subsystems prep — `PreCrash_02.PreCrash_Tueren_Verriegeln=1` (door lock), windows close, sunroof close |
| 4 | TTC ~2.0 s, prewarning | `PreCrash_02.SC_PreCrash_Warnung = 2` ("Vorwarnung"), `AWV_03.FCW_Active = 1` | FCW lamp + brake hydraulic prefill (`AWV_03.Pre_Brake_Fill = 1`) |
| 5 | TTC ~1.4 s, **brake jerk** | `EA_01.EA_Warnruckprofil = 1..7` (one-shot pulse, Profil_1 typical) | Short -3 to -4 m/s² wake-up pulse |
| 6 | TTC ~1.0 s, partial brake | `PreCrash_02.SC_PreCrash_Warnung = 7` ("Basiseingriff") + accel via `ACC_18.ACC_Sollbeschleunigung_02 ≈ -3 m/s²` | ~50% brake force ("Teilbremsung") |
| 7 | TTC ~0.6 s, **emergency brake** | `PreCrash_02.SC_PreCrash_Warnung = 4` ("Eingriff") + `ACC_18.ACC_Sollbeschleunigung_02 ≈ -7 m/s²` (rail to min) | Full brake ("Zielbremsung") |
| 8 | TTC ~0.4 s, takeover request | `PreCrash_02.SC_PreCrash_Warnung = 5` ("Fahreruebernahmeaufforderung"), all `PreCrash_02.PreCrash_*_KSV_verfahren` and `*_Pneumatik_ansteuern` activate | Active head restraints, seat bolster pneumatics, seat reposition |
| 9 | Imminent / impact | `PreCrash_02.PreCrash_Blinken = 3` ("Notbremsblinken") | Emergency brake-light flashing (ESS), brake-lights → hazards transition, stays applied post-impact |
| 10 | Post-impact | `EA_02.EA_Bremslichtblinken = 1`, `EA_02.zFAS_Warnblinken = 1`, `Airbag_01.AB_Front_Crash = 1` latches, `ESC_51.Brake_Pressure` held ~50% until standstill | Post-collision braking; auto-hazards |

Notes:

- TTC thresholds are **Euro NCAP / SSP nominal**; actual values are
  speed-dependent and tuned by the radar.
- VW's `Notbremsblinken` (emergency brake-light flashing) is enabled per
  ECE R48 at decel >7 m/s² + speed >50 km/h.
- The actual brake **command** (force) goes through `ACC_18` (which we
  already TX) — radar's brake request, when relayed, becomes planner's
  accel target. With openpilot longitudinal disabled, radar drives brake
  directly via gateway.
- **Pedestrian-specific FCW**: `PreCrash_02.SC_PreSense_FCWP=1`
  distinguishes pedestrian vs object.
- **Above ~30 km/h** the AEB stack is a **radar+camera fusion** product.
  Below ~30 km/h ("City Emergency Brake") it is **primarily camera** with
  radar confirmation, so VRU detection (pedestrian/cyclist) dominates.
- The **canonical "AEB happening right now"** bit is `VMM_02.AEB_Active`
  (msg 313, bit 31). `Parken_01.AEB_Active` (msg 518, bit 16) is the
  parking ECU's redundant view. `ESC_51.AEB_Breaking_01/02` (8-bit each)
  carry what the ESC is *actually executing* in pressure/torque terms —
  most reliable signal for "did the car brake" vs "did it just warn".

### 7.3 Driver-action gates — when AEB **suppresses or aborts**

Per ECE R152 evasion logic VW must release AEB when the driver acts:

- **Driver brakes hard** (kickdown beyond pedal threshold) → AEB stays
  armed but won't override; jerk/partial brake suppressed.
- **Driver steers >~3 deg/s into a clear lane** → AEB suppressed for evasion.
- **Front Assist disabled in HMI** → soft-off; warns visually but no
  brake. Re-enables on **every key cycle** per ECE — there is no
  persistent off.

### 7.4 Stage signal escalation summary

```
quiet ─── ACC_Optischer_Fahrerhinweis=1 (cluster icon)
          │
          ▼
visual ── AWV_03.FCW_Active=1 + ACC_Akustischer_Fahrerhinweis=1 (chime)
          AWV_03.Pre_Brake_Fill=1 (hydraulic precharge)
          │
          ▼
nudge ─── EA_01.EA_Warnruckprofil>0 + ACC_18.ACC_Sollbeschleunigung_02 dip
          (~250 ms wake-up jerk)
          │
          ▼
brake ─── VMM_02.AEB_Active=1 + ACC_18.ACC_Sollbeschleunigung_02→ -4 m/s²
          ESC_51.AEB_Breaking_01/02>0 (executing)
          PreCrash_02.SC_PreCrash_Warnung=7 (Basiseingriff)
          │
          ▼
emerg.── ACC_Sollbeschleunigung_02→floor (-7.22 / VMM commands ~-9 m/s²)
          PreCrash_02.SC_PreCrash_Warnung=4 (Eingriff)
          PreCrash_02.PreCrash_Tueren_Verriegeln=1 + windows close
          PreCrash_02.SC_PreCrash_Warnung=5 (Fahreruebernahmeaufforderung)
          PreCrash_02 seat-bolster + KSV pneumatic actuation
          PreCrash_02.PreCrash_Blinken=3 (Notbremsblinken / ESS)
          │
          ▼
impact ── Airbag_01.AB_Front_Crash=1 latches
          ESC_51.Brake_Pressure held until v=0
          EA_02.EA_Bremslichtblinken=1 + zFAS_Warnblinken=1
```

### 7.5 `AWV_03` (msg 219, 48 B, sender = front radar)

Currently only **2 of ~250 bits** decoded:

| Signal | Width | Meaning |
|---|---|---|
| `FCW_Active` | 1 b @ 64 | FCW lamp request |
| `Pre_Brake_Fill` | 1 b @ 76 | Hydraulic brake prefill request |

Op10 has richer mapping — most other bytes are `SET_ME_*` placeholders
written as constants (63, 30, 127, 127, 63, 15, 255, 1023, 1) plus
`Pre_FCW (b64\|1)`, `Unknown_01..03`.

There is **no documented brake-target-accel field on `AWV_03`** — it's
the FCW/AEB-indication channel, not the brake-amount channel. **The
actual full emergency brake request lives on the ESC private bus (likely
Flexray on MEB) and is not visible on the gateway PT-CAN.** RE-able only
with a direct ESC controller tap.

**The remaining 48 bytes almost certainly carry**:
- Per-target TTC, range, range-rate
- AEB stage flags (Teilbremsung_Freigabe, Zielbremsung_Freigabe)
- Brake torque/decel command from radar
- Target classification (vehicle / pedestrian / cyclist)

This is the most critical undecoded message for AEB. **Needs raw rlog
frames during AEB events** to crack.

### 7.6 `MEB_AWV_01` (msg 380196013, op10 facelift only)

| Signal | Bits | Meaning |
|---|---|---|
| `Distance_Warning` | 12 \| 1 | "Abstand!" amber chevron |
| `FCW_Display` | 13 \| 1 | red FCW chevron — visual brake-now |
| `AWV_Enabled` | 15 \| 1 | system armed (0 = "Front Assist off" cluster text) |
| `AWV_Init` | 17 \| 1 | uninitialized white icon |
| `FCW_Sound` | 22 \| 2 | `0 silent, 1/2/3 tone profiles` |
| `Target` | 56 \| 8 | object id of triggering target |
| `SET_ME_1` / `SET_ME_511` | constants | cluster handshake |

Not present on MK1 ID.4. For MK1, AEB-HUD likely overlays
`MEB_ACC_01.ACC_Warnhinweis` and `ACC_Optischer_Fahrerhinweis` — RE pending.

### 7.7 `VMM_02` (msg 0x139 / 313, 32 B, gateway)

Vehicle Motion Manager — central long-control arbiter on MEB.

| Signal | Use |
|---|---|
| `AEB_Active` (bit 31) | **TRUE while AEB is autonomously braking** — canonical bit |
| `FCW_Active` (bit 56) | FCW alarm |
| `FCW_Reaction_1/2/3` | escalation steps |
| `Brake_Pressed_1/2/3` | 3 bits of partial→full brake state |
| `ESP_Hold` | held at standstill |
| `Brake_Pressure` (76–86) | independent brake-pressure read |

Has additional warning-state bits beyond `AEB_Active` that change during
the warning period (bits in bytes 4, 5, 7) — should be decoded.

### 7.8 `Parken_01` (msg 0x206 / 518, 24 B)

| Signal | Width | Meaning |
|---|---|---|
| `AEB_Active` | 1 b @ bit 16 | low-speed AEB (parking) engaged — parking ECU's redundant view |

### 7.9 `ESC_51` AEB execution (msg 0x0FC / 252, 48 B)

Already used for wheel speeds and brake pressure. AEB-relevant fields:

| Signal | Width | Meaning |
|---|---|---|
| `AEB_Breaking_01` | 8 b | ESC-side AEB execution channel 1 (target pressure or decel) |
| `AEB_Breaking_02` | 8 b | ESC-side AEB execution channel 2 (likely jerk/phase) |
| `Brake_Pressure` | 9 b @ 0.195 % | total brake pressure (0..50 bar) |
| `HL/HR/VL/VR_Brake_Pressure` | per-corner | per-corner brake pressure (during AEB the front bias is visible) |

These are **what the ESC is executing**, not what the radar requested.
Best signal for "did the car actually brake" vs "did it just warn".

Inactive baseline: `AEB_Breaking_02 = 126`; active offset: `156` (~30
units delta). `AEB_Breaking_01 = 0` baseline; spikes to 200+ are the
wake-up pulse.

### 7.10 `PreCrash_02` (msg 0x13F / 319, 8 B, only bus 1)

Body-CAN message that **executes pre-crash actuation** when AWV says
collision imminent. Op10 does **not send this** — it's stock-only.

| Signal | Width | Enum / Meaning |
|---|---|---|
| `SC_PreCrash_Warnung` | 4 b | **Master AEB stage indicator**: `0 keine, 1 latente_Vorwarnung, 2 Vorwarnung, 3 Akutwarnung, 4 Eingriff, 5 Fahreruebernahmeaufforderung, 6 Abbiegewarnung, 7 Basiseingriff, 8 Heckeingriff` |
| `SC_PreCrash_Texte` | 4 b | Status text (Systemstoerung, keine_Sensorsicht, ESC_aus, …) |
| `SC_PreSense_FCWP` | 1 b | `0=object warning, 1=pedestrian warning` |
| `SC_PreCrash_LED` | 2 b | LED stage: glimmen / leuchten / blinken |
| `PreCrash_Blinken` | 3 b | `0 keine, 1 Warnblinken (hazards solid), 2 RECAS_Blinken, 3 Notbremsblinken (panic flash)` |
| `PreCrash_Tueren_Verriegeln` | 1 b | door-lock request |
| `PreCrash_Schiebedach_schliessen` | 1 b | sunroof-close request |
| `PreCrash_Fenster_schliessen` | 1 b | windows-up request |
| `PreCrash_Anforderung_AFR` | 3 b | adaptive front-restraint pre-fire side: `1 linke_Seite, 2 rechte_Seite, 3 Vorderachse, 4 Hinterachse, 5 Vorwarnung, 7 init` |
| `PreCrash_FS_Pneumatik_ansteuern` (driver) / `_BFS_` (passenger) / `_Fo_` (rear) | 1 b each | seat air-bladder pre-fire (lateral support inflate) |
| `PreCrash_*_Sitzlehne_verfahren` (FS/BFS/Fo) | 3 b each | seatback-position pre-move (forward/back, 6 patterns) |
| `PreCrash_*_KSV_verfahren` (FS/BFS/Fo) | 4 b each | headrest pre-position (Kopfstuetze, active head restraint) |
| `PreCrash_Charisma_FahrPr` | 1..15 | active driving-program profile (Comfort vs Sport vs Eco have slightly different TTC thresholds). Read-only |

**No `PreCrash_Gurtstraffer` field** — belt tension is via
`EA_01.EA_Gurtstraffer_Anf` (EA path), not PreCrash. Pyrotechnic
pretensioners fire from the airbag SG itself based on its own crash
sensors and are not commandable from CAN.

### 7.11 `Airbag_01` (msg 0x40 / 64, 8 B, sub_gateway) — **crash latches + MKB**

Mehrfachkollisionsbremse (MKB) — auto-brake-and-hold after impact to
prevent secondary collision.

| Signal | Width | Meaning |
|---|---|---|
| `AB_MKB_gueltig` | 1 b | MKB armed |
| `AB_MKB_Anforderung` | 1 b | **MKB actively braking after a crash** |
| `AB_Front_Crash` | 1 b | Frontal impact detected, latched |
| `AB_Heck_Crash` | 1 b | Rear impact detected, latched |
| `AB_SF_Crash` | 1 b | Side-front impact `[uncertain on suffix]` |
| `AB_SB_Crash` | 1 b | Side-back impact |
| `AB_Rollover_Crash` | 1 b | Rollover detected `[uncertain]` |
| `AB_Crash_Int` | 3 b | crash intensity 0..7 |
| `AB_EDR_Trigger` | 2 b | event-data-recorder trigger |
| `SC_LowSpeedCrashErkannt` | 2 b | low-speed crash detected |
| `AB_Gurtwarn_VF` / `AB_Gurtwarn_VB` | 1 b each | seatbelt-not-fastened warning, driver / front passenger |

**Important**: there is **no `AB_Gurtwarn_FA`** signal in MEB. Front-row
buckle state is `Airbag_02.AB_Gurtschloss_FA != 3`. Rear-row warnings are
in `Airbag_04.AB_Gurtwarn_Reihe2/3_*`.

These flags drive the **post-crash brake hold** (ESC keeps brakes applied
after impact until standstill). After any of these go high we MUST
disengage and let the car run its post-crash routine.

### 7.12 `Airbag_04` PreSense/AWV configuration mirrors

(`vw_meb.dbc:1529-1548`.) Status/setting mirrors of the FAS suite —
useful for DM to know which features are coded:

- `PreCrash_FAS_Fkt_Status` (VAL 3020): `0 Init, 1 Funktion_Ein, 2 Aus, 3 Fehler`
- `AWV_Einstellung_System_ASG` — AWV (Front Assist) on/off setting
- `AWV_Einstellung_Warnung_ASG` — AWV warn-time (early/medium/late)
- `WarnBrems_Charisma_Status / FahrPr` — warn-brake driving program
- `LGI_FAS_Fkt_Status` — Lane-Guard FAS function status
- `AbstWarn_MV_FAS_Fkt_Status` — distance-warning function status
- `SC_PreSense_Modus_Warnung_NV / _MV` — PreSense warn-modes (forward/medium)

### 7.13 Hazard wake-up — `Blinkmodi_02` (BO 870, `vw_meb.dbc:765-793`)

Output state aggregated from EA + PreCrash requests:

| Signal | Semantics |
|---|---|
| `BM_Crash` | post-crash auto-hazard |
| `BM_Panik` | panic alarm (key-fob) |
| `BM_Not_Bremsung` | **emergency-braking auto-hazard** (auto-flash during AEB/EA) |
| `BM_Warnblinken` | regular hazard switch |
| `BM_Recas` | RECAS rapid-flash (rear-end avoidance signal) |
| `BM_NBA_Status` | NBA = Notbremsblinklicht state machine: `0 nicht_aktiv, 1 BRL_Dunkelphase, 3 BRL_Hellphase` |

`Blinkmodi_02` is **read-only** in op10. The **requests** are
`EA_02.zFAS_Warnblinken` and `EA_02.EA_Blinken=3` from FAS, plus
`PreCrash_02.PreCrash_Blinken` from PreCrash — BCM aggregates them.

### 7.14 Seatbelt pretensioner channels (cross-link to §6)

**Reversible "belt jerk" alert**:

- **`EA_01.EA_Gurtstraffer_Anf`** (b44 \| 2, VAL 2302: `Keine, Haptik_1/2/3`)
  — canonical EA escalation belt-pulse. FAS commands; airbag SG executes
  via **private body sub-CAN** that gateway does NOT mirror to PT-CAN.
  Intent observable; executed pulse is not.
- **`PreCrash_02.PreCrash_Anforderung_AFR`** — Adaptive Front Restraint
  pre-fire on AEB-imminent path. Triggers belt pre-tension AND seat
  positioning together. Not the same as EA reversible-haptic; closer to
  crash-imminent.

**Irreversible (pyro) pretensioners**: airbag SG only on confirmed crash,
not commandable. `Airbag_02.AB_Anprall_*` and `AB_Crashschwere` are the
*post-event* indications.

### 7.15 What the *radar* publishes (ACC_18 vs AEB_Breaking)

We TX `ACC_18.ACC_Sollbeschleunigung_02` ourselves. During a stock AEB
event the radar's ACC_18 keeps publishing comfort-band accel (capped at
`min_accel = -3.5 m/s²`), but the **brake controller**
(`ESC_51.AEB_Breaking_*`) and **VMM_02.AEB_Active** override and produce
the −10 m/s² braking. **Openpilot's panda safety must not block
VMM_02 / ESC_51 / PreCrash_02** — these are read-only on bus 0 and the
brake controller acts on its internal AEB logic regardless.

Confirmed on routes: `volkswagen_meb.h` only restricts TX on `HCA_03`,
`ACC_18`, `MEB_ACC_01`, `LDW_02`, `GRA_ACC_01`. AEB messages are RX-only
and untouched. ✓

The actual AEB brake itself is delivered through the same `ACC_18`
acceleration channel by the radar/MFC (when openpilot is not
long-controlling). When openpilot owns long, the safety panda blocks
`ACC_18` from anyone but us, so true AEB falls back to the EA
emergency-stop path on the radar's failure handler. We currently send
`AWV_03` from openpilot only as a stub in the safety TX list; we do not
generate FCW.

### 7.16 Op10 carcontroller AEB

- `aeb_available = not (PQ flag)` (carcontroller.py:47), so MEB qualifies.
- **All real AEB sending is commented out** at carcontroller.py:227-231.
- AEB messages are **only sent when `DISABLE_RADAR` flag is set**, purely
  as static "AEB off / not initialized" placeholders so the cluster
  doesn't fault when the radar is replaced (lines 246-251). Constant
  `SET_ME_*` payloads.

### 7.17 Op10 radar-disable replacement TX (when `VolkswagenFlags.DISABLE_RADAR` set)

Three messages get spoofed at the panda boundary so the rest of the car
stays calm with stock radar muted:

```
AWV_03      (1 Hz):  SET_ME_63=63, SET_ME_30=30, SET_ME_127=127,
                     SET_ME_127_2=127, SET_ME_63_2=63, SET_ME_15_1=15,
                     SET_ME_255=255, SET_ME_1023=1023, SET_ME_1=1
                     (Pre_FCW=0, FCW_Active=0 by omission)
MEB_AWV_01  (5 Hz):  AWV_Enabled = not disabled (True after first 600 frames),
                     AWV_Init=1, SET_ME_1=1, SET_ME_511=511
Strukturen_01 (25 Hz): empty values dict → all-zero packet
DIAGNOSTIC  (only):  UDS tester-present \x02\x3E\x80\x00\x00\x00\x00\x00
```

Without `DISABLE_RADAR`, openpilot only **RX**-es `AWV_03.FCW_Active` for
`ret.stockFcw`. **Crucially, openpilot must NOT send these spoofs**
without that flag, or panda's relay-check rejects.

### 7.18 What we observed on routes

| Counter | stock-acc | flicker | heavy-override |
|---|---|---|---|
| `AWV_03.FCW_Active` rises | 0 | 0 | 0 |
| `VMM_02.FCW_Active` rises | 0 | 0 | 0 |
| `VMM_02.AEB_Active` rises | 0 | 0 | 0 |
| `Pre_Brake_Fill` frames | 0 | 0 | 0 |
| `PreSense_FCWP` frames | 0 | 0 | 0 |
| `ACC_Warnhinweis` frames | 0 | 0 | 0 |

**No FCW/AEB activity in any of the 4 reference routes.** Same situation
as EA — to capture progression we need a stock route where AEB actually
fires.

| Route | Time | Msg | Bytes | Decoded |
|---|---|---|---|---|
| f4 | 102.7 s | EA_01 | `0c0400000002c0ff` | CTR=4, EA_Funktionsstatus=**2 STANDBY**, Warnruck=0, Gurtstraffer=0, Accel=neutral. Quiet state |
| f4 | 266.9 s | VMM_02 | `4dbd141001001022…` | New byte pattern at `byte4=0x10, byte5=0x01, byte7=0x22` (vs baseline `0x00 0x00 0x20`). New flags asserted but `AEB_Active(bit31)=0`. Likely visual-only warning |
| f3 | 308.0 s | **AWV_03** | `1b31005efefe0000fc0ffdc0…` | **FCW_Active = 1, Pre_Brake_Fill = 1** — Forward Collision Warning fired with hydraulic precharge. The 48-byte payload also shows `byte 3 = 0x5E`, `byte 4–5 = 0xFE 0xFE`, `byte 8–11 = 0xFC 0x0F 0xFD 0xC0` — undecoded TTC/target fields |
| f3 | 205.3, 246.4, 287.4 s | VMM_02 | `4dbd141001001022…` | Same warning-mode pattern; sustained ~80 s. Correlates temporally with the FCW event at 308 s |
| both | various | Airbag_01 | `…44c08017XX` | byte 2 = 0x44 (no crash). byte 7 increments — looks like a heartbeat counter |
| both | full route | VMM_02.AEB_Active | bit 31 | **Never set to 1.** No actual AEB intervention occurred — only FCW warning |

**Confirms**:
1. `AWV_03.FCW_Active` and `Pre_Brake_Fill` are correctly decoded and
   they fired at the FCW moment.
2. `VMM_02` has additional warning-state bits beyond `AEB_Active` that
   change during warning period (bytes 4, 5, 7).
3. Full 48-byte `AWV_03` payload contains the radar's TTC and target
   classification — bytes 8–11 are non-zero during FCW. These are the
   missing fields.
4. `PreCrash_02` rarely fires — only 1 frame across both routes, and it
   was on bus 1.


---

## 8. Stock-route capture plan — what's enough to replicate 1:1?

**Short answer (post-f3/f4 analysis):** the f3+f4 routes captured PHASE2 and
PHASE3 escalation precisely (timing, codes, actuators), and **immediately
invalidated 5 theoretical assumptions** (see §6.14). For the **post-stop SOS
chain** (full deceleration to vEgo == 0, automatic horn, eCall, door unlock,
brake-light flash), f3 and f4 are not enough because the driver always
recovered before standstill.

**One additional route where the driver lets the car stop completely** is the
single biggest gap.

### 8.1 What ONE stock route gives you

For each *escalation chain*, a recorded route gives you:

1. **Exact timing** — phase durations, warning chime timing, jerk-profile
   cadence, prefill-to-AEB latency. Not in the DBC.
2. **Exact codes** — which `EA_Texte`, `ACC_Texte_Primaeranz_02`,
   `ACC_Texte_Zusatzanz_02`, `EA_Warnruckprofil`, `EA_Gurtstraffer_Anf`
   value the system actually uses for each phase (vs. DBC enum which lists
   *possible* values).
3. **Conditional gating** — at what speed PHASE 2 vs PHASE 3 is allowed,
   whether eCall fires only after certain duration, etc.
4. **Cross-message synchronization** — exact ordering of
   `EA_Funktionsstatus` ↔ `EA_Texte` ↔ `ACC_Akustischer_Fahrerhinweis` ↔
   `EA_Gurtstraffer_Anf`.
5. **Regression baseline** — replay = ground-truth diff target for our
   generated CAN.

Plus the constant-rate plumbing — **EA_STANDBY** baseline values (so you
know what "no event" looks like for every signal, byte-by-byte),
**HCA_03 / ACC_18 / GRA_ACC_01** stock cadence + signal patterns when factory
TA is engaged (gold-standard reference for matching openpilot output),
KLR_01 normal touch values + Touchauswertung distribution, BSM events if
traffic appears in blind spots, Distance_Status / radar-track lifecycles,
buttons & cruise-state transitions.

### 8.2 What a stock route does NOT give you

1. **Causality** — recording shows A precedes B but not whether A causes B.
   Mitigation: comma device captures all 3 buses, so ECU origin is
   distinguishable.
2. **Actuator response to OUR commands** — if we publish `EA_Texte = 3` on
   bus 0, will the cluster show it? Probably yes (it listens to bus 0 for
   forwarded messages), but only HIL on the actual car proves it. The
   radar/brake/SRS ECUs may sign or counter-protect their own outputs
   (CRC + 4-bit BZ counter), meaning we cannot impersonate the radar
   without keeping the counter sequence intact.
3. **Negative space** — the route shows what *did* happen, not what *would*
   happen with different driver behavior or threat geometry. Multiple
   routes needed to isolate variables (lane-change-induced false-AEB
   suppression, low-speed cutoff, etc.).
4. **Long-term fault recovery** — `EA_Funktionsstatus = 7/8`
   (reversible/irreversible fault) progression rarely seen organically.
5. **AWV_03 contents (48 B)** — only 2 of ~250 bits decoded. TTC, target
   class, brake-jerk command live here. Even with rlog, would need
   **multiple recordings of the same scenario** at different speeds and
   target distances to triangulate which bits encode TTC vs decel.
6. **MEB_Camera_01 / 02** — 64-byte messages with mostly NEW_SIGNAL_x.
   May be where the camera publishes lane-departure-warning-derived
   take-over gating. Same multi-recording problem.
7. **MEB_Side_Assist_02 (64 B)** — internal track list. Need controlled
   parking-lot scenarios with cross-traffic.
8. **AEB full-brake command** — lives on ESC private/Flexray, not
   gateway-visible.
9. **`KLR_Touchauswertung` numeric semantics** (op10 hardcodes 10; need
   hands-on/off transitions to map exact threshold).
10. **MEB `LDW_Texte` codes 1, 3, 6, 7, 9, 10** — only 4 and 8 are in the
    DBC enum; MQB suggests the others but unverified on MEB.
11. **`AWV_03.Unknown_01/02/03`, `MEB_AWV_01.Target`** — unknown semantics.
12. **Whether the gateway/airbag SG accept foreign senders for `EA_01`,
    `PreCrash_02`, `EA_Gurtstraffer_Anf`** — untested. If not, we cannot
    send those messages to drive the body modules; we can only consume
    them.

### 8.3 Concrete recording plan (one route per row, ~5 minutes each)

| # | Goal | Status | Driver script |
|---|---|---|---|
| 1 | EA PHASE 2 + PHASE 3 escalation (recovered) | **DONE** (`f3`, `f4`) | engage TA, drop hands, recover at PHASE 3 |
| 2 | EA full SOS stop (vehicle to standstill) | TODO — **#1 priority** | engage TA on a safe road @60 km/h, drop hands cold, **do not intervene** until the car has stopped and waited 10 s. Captures: post-stop deceleration profile, brake-light flash, hazards mode, door unlock, interior light, eCall, automatic horn. Also `EA_Anforderung_HMS = 1..6` (halten/parken/anfahren), `MFL_Signalhorn`/`SMLS_Hupe`, `EA_eCall_Anf = 1`, `zFAS_Warnblinken`, `EA_Tueren_Anf`, `EA_Innenraumlicht_Anf`, `EA_Sollbeschleunigung` profile during controlled-stop. **Pull SIM / airplane-mode head unit** so eCall doesn't dial. *Optional but ideal:* also capture **PHASE0 → PHASE1 graded warnings** (skip on f3/f4) by driving normally with TA engaged, ONE hand off (light touch elsewhere) and let icon escalate slowly without triggering PHASE2 |
| 3 | EA capacitive partial-grip | TODO | engage TA, rest fingertips (no torque) — see whether `KLR_Touchintensitaet > X` keeps EA in STANDBY (avoids escalation) |
| 4 | DM camera Sekundenschlaf (drowsiness) | optional | close eyes ≥3 s with TA engaged. Captures `EA_Texte=9`. May share PHASE2 entry with #2 |
| 5 | FCW + AEB pre-fill | TODO | controlled approach to a parked inflatable target @60 km/h, no driver intervention. Captures `AWV_03.FCW_Active`, `Pre_Brake_Fill`, `ESC_51.AEB_Breaking_01` ramp, `VMM_02.AEB_Active`, `MEB_ACC_01.ACC_Wunschgeschw_Farbe` red change, `PreCrash_02.SC_PreCrash_Warnung 0→1→2→3` |
| 6 | FCW + partial brake against soft target | TODO | needs actual lead vehicle or controlled-test foam target. Approach at ~30 km/h, brake before contact. Distinguishes FCW-only from partial-brake by whether `Pre_Brake_Fill` flips |
| 7 | AEB full brake | optional, only if safe target | same scenario but allow full intervention. Captures high end of `AEB_Breaking_01/02`, brake-light blinking, `PreCrash_02.SC_PreCrash_Warnung` rising 1→2→3→4 |
| 8 | LKAS takeover | TODO | engage TA on a curve, override with steering, observe `EA_Texte = 10/11` |
| 9 | drowsiness | optional | attempt `EA_Texte = 9` via slow lane drift |
| 10 | Set-speed adjustment | TODO | while ACC engaged, repeatedly tap UP and DOWN on the stalk. Captures COUNTER advance pattern + `ACC_Wunschgeschw_02` response so we calibrate button-injection rate |
| 11 | Driver gas override at standstill behind lead | TODO | record stock car doing it; compare braking jerk profile against our `accel_last`-based limiter |
| 12 | Lane-change with BSM warning | TODO | trigger turn signal while car overtaking adjacent lane; captures `Blind_Spot_Warn_*` rising and `Blind_Spot_*` intensity ramp |

**Skip / cannot capture safely**:

- **Full AEB / pyro / actual crash** — Pyro pretensioners are one-shot
  (destroys parts). Full AEB uses ESC private bus not visible on PT-CAN.
  Do not attempt.
- **Post-collision MKB** — `Airbag_01.AB_MKB_Anforderung=1`,
  `AB_Front_Crash=1`, `ESP_20 Notbremsung_aktiv`. Cannot be safely staged.
  Source from a service-shop incident log if available, or a comma
  engineer's existing MKB log.
- **PreCrash D** — at higher closing speed where PreCrash arms (~50+ km/h
  relative). Hard to test safely without target sled.

If you can only record **one**, prioritize **#2 (full EA escalation 0→3)**
— it exercises the most actuators (jerk, belt, hazards, brake-light flash,
doors, eCall) in one go and is what we'd otherwise need a CAN simulator for.

### 8.4 Recommended minimum

- **#1 (f3/f4) + #2 (full EA SOS)** — complete EA characterization
- **#5 + #6 (FCW probes)** — AEB stage characterization
- **#11 + #12** — port validation

Crash-only signals (`AB_Front_Crash`, `AB_MKB_Anforderung`,
`PreCrash_02.SC_PreCrash_Warnung`) cannot safely be captured in a stock
route. We just trust the safety panda to let the radar through unmodified.

### 8.5 What ONE route gets wrong even if heroic

- Timing thresholds (10 s → 20 s → 25 s → 35 s) are coded into the camera
  and **vary by speed and feature trim**. A single capture gives one
  (speed, trim) tuple. Replicating "1:1" implies replicating the ECU
  logic, not just the bus signals — and the ECU logic is closed-source.
- Some rare states (`EA_Funktionsstatus=7` reversibler_Fehler, `=8`
  irreversibler_Fehler, `EA_Texte=14/15`) only fire under fault injection.
  You won't see them unless something breaks.

### 8.6 KLR (capacitive wheel) trim caveats

- KLR trims and non-KLR trims escalate on different timers. f4 and f3 are
  on the same car, so timings are consistent with each other but may not
  transfer to non-KLR ID.4.
- `EA_Funktionsstatus` PHASE0/PHASE1 may exist on different VW software
  levels — these routes show STANDBY → PHASE2 → PHASE3 directly. Older
  MY22 firmware may have additional steps.

### 8.7 Capture protocol

For each scenario, ideally:

- `rlog.bz2` (full CAN logs)
- A timestamp of the moment each stage triggered (read it off the cluster
  — "now I see icon", "now I hear chime", "now I felt the seatbelt jerk")

With those, we can:
1. Replay rlog through cabana
2. Filter to relevant message IDs
3. Plot each unknown bit/byte
4. Correlate transitions with timestamps
5. Decode the missing fields in `vw_meb.dbc`

After that, we can **mirror OEM behavior** in openpilot: replicate
wake-up brake jerk via `ACC_18.ACC_Sollbeschleunigung_02` pulses, use
`MEB_Distance_01` lead data + our own state machine for AEB stages,
surface `EA_Funktionsstatus`-like progression to openpilot HUD.

### 8.8 Safety caveats

- Scenarios 1–4 require a **closed road or empty lot**. Hands-off +
  waiting for full EA stop on a public road is illegal and dangerous in
  most jurisdictions.
- Scenarios 5–7 need a **controlled lead** (foam barrier on a closed
  track, or stock VW dealer test rig). Don't approach real cars at speed.
- Recording from the **panda alone** without OP active is preferred so
  the stock stack runs unmodified.


---

## 9. On-route signal verification — route `000000d8` (124 k frames, ≈30 min highway)

| Message | Bus | Hz | Notes |
|---|---|---|---|
| `AWV_03` (0x0DB) | 1 + 2 | 1.0 | present, FCW_Active stayed 0 |
| `HCA_01` (0x126) | 2 | 1.0 | EA_ACC_Sollstatus = 0 entire drive |
| `MEB_Camera_01` (0x183) | 2 | 25.0 | front camera scene, signals largely unreversed |
| `EA_01` (0x1A4) | 1 + 2 | 2.0 | Funktionsstatus = 2 (STANDBY) all 184 k frames |
| `EA_02` (0x1F0) | 2 | 2.0 | EA_Texte = 0 throughout |
| `KLR_01` (0x25D) | 0 | 16.7 | Touchauswertung = 0 (60 %) and 7 (40 %); intensities up to 36/33/41 of 250 |
| `PreCrash_02` (0x13F) | 1 | 5.0 | warn level 0 throughout |
| `Motor_41` (0x2C2) | 1 | 2.0 | PEA_Texte = 0 throughout |
| `VMM_02` (0x139) | 0 | 50.0 | AEB_Active = 0 throughout |
| `Parken_01` (0x206) | 1 | 50.0 | parking-AEB = 0 throughout |
| `ESC_51` (0x0FC) | 0 | 100.0 | AEB_Breaking_01/02 = 0 throughout |

KLR_01 fails CRC with `vw_meb` checksum (cross-ref §6.2). Read-only is
fine, but TX needs per-message CRC override.

---

## 10. Bus topology / safety

This car (J533 gateway harness):

| Bus | Source | Notes |
|---|---|---|
| **0** (`Bus.pt`)  | gateway → powertrain | what panda's safety hooks see. RX: `ESC_51`, `Motor_51`, `Motor_14`, `LH_EPS_03`, `QFK_01`, `Gateway_72/73`, `Airbag_01`, `LWI_01`, `Blinkmodi_02`, `GRA_ACC_01`, `EM1_*`, `MEB_HVEM_*` |
| **1** (`Bus.alt`) | extended/radar | `PreCrash_02` was seen *only* on bus 1 in our routes |
| **2** (`Bus.cam`) | camera + radar + EA module | `MEB_Side_Assist_01/02`, `RCTA_01`, `MEB_Camera_01/02/03`, `EA_01/02`, `KLR_01`, `TA_01`, `AWV_03`, `MEB_AWV_01`, `LDW_02`, `MEB_ACC_01`, `MEB_Distance_01` (renamed `Strukturen_01` upstream), and same powertrain RX echoed |
| 0 | TX (when relay closed) | openpilot's `HCA_03`, `LDW_02`, `ACC_18`, `MEB_ACC_01`, optional `EA_01/02`, `KLR_01`, `TA_01` — what panda forwards to powertrain |
| 128 | TX echo | openpilot's TX sniffed back — confirm what we sent |
| 192 | TX rejected | safety blocked — what we tried to send and panda refused |

Bus 0 / Bus 2 source convention: gateway-forwarded duplicates appear on
both. EA_01 and TA_01 emit on **bus 0, bus 1, AND bus 2** simultaneously
with identical content — panda sees them regardless of relay side.

### Safety / vehicle-model alignment

`volkswagen_meb.h` uses `steer_angle_cmd_checks_vm` with real VM params:
```c
.slip_factor = -0.0006055171512345705f,
.steer_ratio = 15.6f,
.wheelbase   = 2.77f,
```
`angle_deg_to_can = 10` (CAN units of 0.1 deg of steering wheel).
`max_angle = 6000` (600 deg, well above the rack lock-to-lock).

Conversion path each TX (cross-ref §1.1).

---

## 11. Key file:line references (op10 + DBCs)

### Current DBC

- `vw_meb.dbc:171-175 AWV_03`
- `vw_meb.dbc:279-293 HCA_01`
- `vw_meb.dbc:352-374 PreCrash_02`
- `vw_meb.dbc:455-464 EA_01`
- `vw_meb.dbc:466-478 EA_02`
- `vw_meb.dbc:525 MEB_Distance_01`
- `vw_meb.dbc:560-570 KLR_01`
- `vw_meb.dbc:583-601 Motor_41`
- `vw_meb.dbc:765-793 Blinkmodi_02`
- `vw_meb.dbc:840-860 LDW_02`
- `vw_meb.dbc:1497-1527 Airbag_02`
- `vw_meb.dbc:1529-1548 Airbag_04`
- VAL_ definitions (~lines 2230-3020)

### Facelift DBC (op10)

- `vw_meb_2024.dbc:2078-2086 MEB_AWV_01`

### Op10 carcontroller

- `47 aeb_available`
- `153-160 HCA EA torque spoof`
- `163-171 KLR spoof send`
- `178-183 blinker EA forward`
- `227-231 AEB commented`
- `246-254 radar-disable AEB-fake`
- `260-269 LDW HUD`

### Op10 mebcan

- `38 create_eps_update`
- `56-83 create_blinker_control`
- `86-107 create_lka_hud_control`
- `128-145 create_capacitive_wheel_touch`
- `339-358 create_aeb_control`
- `361-369 create_aeb_hud`

### Op10 carstate

- `290-292 EA/KLR consume`
- `312 seatbelt`
- `326 LDW stock`
- `328 stockFcw`
- `385-386 blinker active`
- `404-405 EA stock dicts`
- `552-572 MEB get_can_parsers`

### Op10 values flags

- `249-253 STOCK_HCA_PRESENT=1, STOCK_KLR_PRESENT=64`

---

## 12. Quick reference — key signals

```
HCA_03         0x303  TX  steering control
ACC_18         0x14D  TX  longitudinal accel command
MEB_ACC_01     0x300  TX  ACC HUD (set-speed, lead, alerts)
LDW_02         0x397  TX  LDW HUD (LED bar, takeover text)
GRA_ACC_01     0x12B  RX  cruise stalk + cancel-spam TX

QFK_01         0x13D  RX  curvature, EPS state machine
LH_EPS_03      0x09F  RX  driver torque
ESC_51         0x0FC  RX  wheel speeds, brake pressure, AEB pressure
ESC_50         0x...  RX  yaw rate (Yaw_Rate / Yaw_Rate_Sign)
Motor_51       0x10B  RX  TSK_Status, accel pedal pressure
Motor_14       0x3BE  RX  brake-pedal switch
Motor_41       0x2C2  RX  PEA_Texte
Gateway_73     0x...  RX  gear shifter, EPB
Airbag_02      0x...  RX  driver belt buckle (AB_Gurtschloss_FA)
Airbag_01      0x040  RX  crash latches + MKB
Airbag_04      0x...  RX  PreSense/AWV configuration mirrors

MEB_Side_Assist_01   0x24C  RX (cam)  blind spot — IMPLEMENTED
MEB_Side_Assist_02   0x24D  RX (cam)  unknown — RE candidate (raw side-radar tracks)
RCTA_01              0x2B7  RX (cam)  rear cross-traffic alert — only CRC + counter
MEB_Distance_01      0x24F  RX (cam)  fused 6-object track list — IMPLEMENTED

EA_01          0x1A4  RX (cam)  EA phase, jerk profile, belt pretensioner, EA accel
EA_02          0x1F0  RX (cam)  EA texts, hands-off lamp, hazards, brake-flash
TA_01          0x26B  RX (cam)  Travel Assist status / availability
KLR_01         0x25D  RX (pt)   capacitive wheel touch — facelift / equipped wheel only
HCA_01         0x126  RX (cam)  EA → ACC handoff signals + EPS request

AWV_03         0x0DB  RX (cam→pt)  FCW + Pre-Brake-Fill (front radar)
MEB_AWV_01     0x16A954AD          AEB HUD (op10 facelift only)
VMM_02         0x139  RX (pt)      AEB_Active / FCW_Active / brake pressure
PreCrash_02    0x13F  RX (alt)     SRS pre-crash actuator chain
Parken_01      0x206  RX (cam)     low-speed parking AEB
ESP_20         0x659  RX           Notbremsung_aktiv

TM_01          0x5A7  RX           TM_Nur_Hupen — telematics auto-honk candidate
MFL_01         0x331  RX           MFL_Signalhorn — driver wheel-button horn
SMLS_01        0x3D4  RX           SMLS_Hupe — alarm-system horn chirp
```

---

## 13. Open work / implementation status

### 13.1 Status table (this branch / consolidated across ports)

| Item | Status | Notes |
|---|---|---|
| `update_hca_state_meb` | ✓ | Aligned to op10 verbatim — `DISABLED` + sustained `FAULT` after `eps_init_complete` = permanent |
| `create_lka_hud_control` | ✓ | LDW_Texte = `laneAssistTakeOver` from `VisualAlert.steerRequired`; `display_mode = 1 if lat_active` (travel-assist yellow lanes) |
| Side radar (BSM) parser wiring | ✓ | this PR |
| Jerk-limited ACC_18 (override-end smoothing) | ✓ | ±2.0 m/s³ ISO comfort, op2 7-line patch |
| `steer_angle_cmd_checks_vm` panda safety | ✓ | real ID.4 bicycle VM, 600 deg max angle |
| 100% safety coverage on `volkswagen_meb.h` | ✓ | 63 MEB tests, 3157 total safety tests |
| Front-radar parser (`MEB_Distance_01`) | ✓ | 6 fused tracks, 25 Hz, op1 verbatim copy |
| `radarUnavailable = False` for MEB | ✓ | gated on `0x24F in fingerprint[0]` |
| `Bus.radar: 'vw_meb'` in dbc_dict | ✓ | enables radar parser |
| `update_steering_pressed(condition, 5)` hysteresis | ✓ | 50 ms debounce (Tesla/Ford/PSA pattern) |
| `ret.yawRate` from `ESC_50` | ✓ | fixes `_check_saturation` undershoot detection |
| Dashcam mode | ✗ removed | full stable port; CARS.md regenerated → "Upstream" |
| Route replay test | ✓ | 4 routes, 0 false EPS faults, BSM validated |
| `chassis_codes` += `"E8"` | ✗ | **TODO before merge** — required for US ID.4 |
| WMIs += `{USA_SUV, EUROPE_CAR}` | ✗ | TODO — unblocks US/EU |
| Set-speed forwarding (`Tip_Hoch`/`Tip_Runter`) | ✗ | follow-up; not needed for minimal port |
| Side-radar tracks (`MEB_Side_Assist_02`) reverse-engineering | ✗ | follow-up; only 2 / unknown signals decoded |
| EA mirror (visual + audible only) on `MEB_ACC_01` | ✗ | follow-up; requires recorded EA escalation route |
| AEB pass-through validation | inherent | RX-only on our buses, panda doesn't gate |
| `STOCK_KLR_PRESENT` pass-through | ✗ | cap-wheel trims time out stock TA on top of openpilot without us broadcasting `KLR_01`. Port op10's `mebcan.create_capacitive_wheel_touch` (~10 lines). Plus implement MQB-style CRC for KLR_01 in safety + packer |
| MEB GEN2 / MK2 platform | ✗ | low risk, mostly mechanical work (DBC + flag + RxCheck lengths). Estimated 1 day of porting + re-running tests |
| `RCTA_01` payload | ✗ | not yet decoded in either DBC. RCTA reaches driver via PDC speakers (`PDC_Tonausgabe_*`) but trigger state lives on `RCTA_01` |
| RCTA HUD passthrough | ✗ | needs cereal RadarData extension or custom alert |
| EA escalation passthrough → openpilot HUD | ✗ | needs the full SOS stock-route capture |
| KLR_01 hands-on spoofing TX | ✗ | needs MQB-style CRC implementation in safety + packer |

### 13.2 Open follow-ups (priority order)

1. **Process replay routes through the new carcontroller** to confirm
   end-to-end that the LKAS-fault on 000000ed no longer reproduces and
   that the smooth-brake handoff settles to comfort jerk.
2. **`STOCK_KLR_PRESENT` pass-through.** Cap-wheel trims will time-out
   stock TA on top of openpilot without us broadcasting `KLR_01`. Port
   op10's `mebcan.create_capacitive_wheel_touch` (~10 lines).
3. **Set-speed button-tap state machine** — most-asked feature. ~80 LoC.
4. **MEB GEN2 / MK2 platform** — low risk, mostly mechanical work
   (DBC + flag + RxCheck lengths).
5. **`MEB_Side_Assist_02` reverse-engineering** — almost certainly
   carries per-zone radar tracks like `MEB_Distance_01` does for the
   front. Worth a sniffing campaign with a lateral-moving target.
6. **`RCTA_01` payload** — not yet decoded in either DBC.
7. **DM passive read** — surface `EA_Funktionsstatus`,
   `ACF_Lampe_Hands_Off`, `EA_Texte` in carstate so openpilot can mirror
   the cluster's own escalation in its UI.
8. **AEB passive read** — surface `AWV_03.FCW_Active` → `ret.stockFcw`,
   `VMM_02.AEB_Active` → carstate flag.
9. **AEB radar-replacement TX** — only if `DISABLE_RADAR` becomes a
   supported flag.
10. **ID.5 docs entry** — 1-line change.
11. **KLR spoof port** — gated on `STOCK_KLR_PRESENT`, only useful if
    user has the capacitive wheel and wants to defeat the hands-off
    escalation while running openpilot.
12. **Raw front-radar tracks via UDS** — expected negative result.

### 13.3 Open work effort estimate

| # | Item | LOC est. | Blocker |
|---|---|---|---|
| 1 | Set-speed up/down button injection | ~30 | none — ready to implement |
| 2 | MK2 / GEN2 platform support | ~50 + DBC import | needs `vw_meb_2024.dbc` copied from sunnypilot (license OK, MIT) |
| 3 | Radar interface enable | ~10 + safety RX checks | `radar_interface.py` already drafted ✓ |
| 4 | RCTA HUD passthrough | ~20 | needs cereal RadarData extension or custom alert |
| 5 | EA escalation passthrough → openpilot HUD | ~40 | needs the stock-route capture #2 |
| 6 | KLR_01 hands-on spoofing TX | ~30 | needs MQB-style CRC implementation in safety + packer |
| 7 | MEB_Side_Assist_02 reverse engineering | unknown | needs route with overtaking traffic |

---

## 14. Open questions / future work (uncategorized)

- ID.4 MK1 facelift (2023+ "Performance" trim) — does it use MK1 DBC or
  MK2 DBC? Need a VIN sample to confirm.
- Whether ID.5 (E2 chassis, same as ID.4) has any wheelbase/mass divergence
  worth a separate platform vs. tagging both under `VOLKSWAGEN_ID4_MK1`'s
  CarDocs.
- Whether the cluster offset on this user's car is large enough to need
  the kmh/mph + cluster offset path (set-speed Path C in §2).
- Whether ID.4's stock TA progression includes a reversible pretensioner
  — reachable from CAN or only on private body bus.
- MK2 vs SSP timing — meb6 says no MEB MK2 ID.4 yet (SSP starts ~2026/27),
  others describe MK2 as a real facelift. Need a MK2 VIN sample to settle.

---

## 15. Sources

### Local

- `opendbc/opendbc/dbc/vw_meb.dbc` — primary DBC reference
- `opendbc/opendbc/car/volkswagen/carstate.py` — current usage
- jyoung8607's PRs against opendbc that contributed the EA/PreCrash decodes

### Public (consulted via research agents)

- VW Self Study Programs — SSP 890253 (MQB driver assistance), SSP 890433+
  (MEB), SSP 626 (driver-monitoring/EA progression)
- jyoung8607/openpilot vw-meb branch wiki and issues
- spot2000/Volkswagen-MEB-EV-CAN-parameters — UDS PID inventory
- evflux.pro MEB pages (sensor lists, ECU networking)
- Euro NCAP test reports for ID.3 / ID.4 — measured AEB TTC and decel
- ID Talk / vwidtalk forums — driver-experienced Travel Assist progressions
- sunnypilot  — set-speed and DM handling patterns

### Note on uncertainty

Where this document says `[uncertain]` or "theoretical", the claim is
reasonable from public signals but not personally verified against a
production fleet or an ECE-spec'd traceback. Treat exact stage-timing
numbers as nominal — the user's f3/f4 stock-route captures replaced these
with measured values where available; future SOS-to-standstill capture
will close the remaining gaps.

