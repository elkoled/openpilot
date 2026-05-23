# lateral_fleet

Fleet-scale lateral / EPS diagnostic for VW MEB (initially VOLKSWAGEN_ID4_MK1).

The tool extracts per-segment residuals from openpilot logs, pools them
into a bucketed K(v, c) table, and fits a parametric K(v) plant model
across dongles. It deliberately reports both empirical and parametric
views and quarantines K outside (0, 1] rather than letting overfit
results through.

## Dependencies

```
uv pip install -r tools/lateral_fleet/requirements.txt
```

`matplotlib` is already in openpilot's `dev` extras; `pandas` and
`pyarrow` are not.

## CLI

```
python -m tools.lateral_fleet ingest      --routes routes.csv      --workers 16
python -m tools.lateral_fleet ingest-csv  --csv    eps_seglist.csv --workers 16
python -m tools.lateral_fleet aggregate   --run    runs/<ts>/
python -m tools.lateral_fleet plant       --run    runs/<ts>/
python -m tools.lateral_fleet report      --run    runs/<ts>/
python -m tools.lateral_fleet hypothesis  --run    runs/<ts>/ --name per_vin_scalar
```

`ingest-csv` expects the eps_seglist.csv schema:
`segment, dongle, platform, mean_v, max_v, engaged_frac, rlog_url`.

Both ingest variants are incremental — segments whose status sidecar reads
`ok` are skipped on re-runs. `--retry-failed` retries quarantined segments.

## Pooling

Default is **hierarchical**: per-dongle K(v, c) means first, then unweighted
mean across dongles for the fleet estimate. CIs are bootstrap over
dongles. Use `--weighted` to fall back to count-weighted pooling (which
lets a chatty dongle dominate the fleet number).

## Sanity guards

- Cross-dongle aggregation operates on per-dongle pooled estimates, never
  raw routes. A single chatty dongle cannot dominate a fleet plot.
- Any fit returning a passive-plant gain `K ∉ (0, 1]` is quarantined.
- Cells with fewer than `MIN_DONGLES` per bucket are rendered as
  "insufficient", not as a point estimate.
- CAN decode goes through `opendbc.can.CANParser` against
  `opendbc_repo/opendbc/dbc/vw_meb.dbc`. No hand-rolled byte slicing.
- Held-out splits for any cross-validated hypothesis use **dongle** as the
  split key, not route.
- Sign of `c_yaw` empirically verified against `c_eps` on
  f73c01590368ee5b/0000000e--2d623b6df3 (sunnypilot PID reference route);
  see `features.curvature_from_yaw`.
