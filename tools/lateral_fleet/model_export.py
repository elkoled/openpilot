"""Export the fleet K(v) model to a portable JSON + a self-contained
Python module that can be imported by openpilot's carcontroller without
pulling in pandas/pyarrow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from openpilot.tools.lateral_fleet import model_apply


_PY_TEMPLATE = '''"""
VW MEB ID4 fleet EPS plant model -- K(v) lookup.

Generated from {n_dongles} engaged-driving ID4 dongles, {n_routes} segments,
hierarchical (per-dongle then unweighted-mean-across-dongles) pooling.
CIs are 5-95 percentile bootstrap over dongles.

K(v) is the steady-state plant gain: actual_curvature = K(v) * commanded_curvature
fit through-origin on samples with |c_cmd| >= {min_cc:g} rad/m.

Caveats:
 * Per-dongle dispersion at 120 km/h is wide (std={std_120:.2f}); the
   correction at that speed is less reliable than at 60-100 km/h.
 * Below {first_anchor:.0f} km/h the data is sparse; we return K=1.0
   (passthrough).
 * Yaw-derived K is what determines lateral tracking; EPS-rack-derived K
   is higher (the EPS does more than the vehicle yawss responds to).
 * This is open-loop steady-state. Plant lag (~{lateral_delay:.2f} s) is
   handled separately by openpilot's liveDelay; do not modify it here.
"""
from __future__ import annotations

# (speed_kph, K_mean, K_lo, K_hi, n_dongles_valid)
FLEET_K_YAW_VW_ID4_MK1 = [
{rows_yaw}
]

FLEET_K_EPS_VW_ID4_MK1 = [
{rows_eps}
]


def K_at_speed(v_ego_ms: float, table=FLEET_K_YAW_VW_ID4_MK1, min_dongles: int = {min_dongles}) -> float:
  """Return K at v_ego (m/s) via linear interpolation. Returns 1.0
  (passthrough) below the lowest anchor, for insufficient cells, or
  outside the valid range."""
  v_kph = float(v_ego_ms) * 3.6
  if v_kph < table[0][0]:
    return 1.0
  if v_kph > table[-1][0]:
    K = table[-1][1]
    return K if K and K > 0 else 1.0
  # Linear interp on valid neighbors.
  for i in range(len(table) - 1):
    v0, K0, _, _, n0 = table[i]
    v1, K1, _, _, n1 = table[i + 1]
    if v0 <= v_kph <= v1:
      if n0 < min_dongles or n1 < min_dongles:
        return 1.0
      t = (v_kph - v0) / (v1 - v0)
      K = K0 * (1 - t) + K1 * t
      return K if K and K > 0 else 1.0
  return 1.0


def apply_correction(c_cmd: float, v_ego_ms: float, max_boost: float = 2.0) -> float:
  """Boost commanded curvature by 1/K(v) to compensate for plant
  undershoot, capped at `max_boost`."""
  K = K_at_speed(v_ego_ms)
  if K <= 0:
    return float(c_cmd)
  boost = 1.0 / K
  if boost > max_boost:
    boost = max_boost
  return float(c_cmd) * boost
'''


def export_model(run_dir: Path, out_py: Path | None = None, out_json: Path | None = None) -> dict:
  run_dir = Path(run_dir)
  yaw = pd.read_parquet(run_dir / 'plant_fleet_yaw.parquet').sort_values('speed_kph')
  eps = pd.read_parquet(run_dir / 'plant_fleet_eps.parquet').sort_values('speed_kph')
  dongles = pd.read_parquet(run_dir / 'dongles.parquet')
  routes = pd.read_parquet(run_dir / 'routes.parquet')
  per_dongle = pd.read_parquet(run_dir / 'plant_fit_yaw.parquet')

  def _rows(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.iterrows():
      out.append(f"  ({r['speed_kph']:.1f}, {r['K_fleet_mean']:.4f}, "
                 f"{r['K_fleet_lo']:.4f}, {r['K_fleet_hi']:.4f}, "
                 f"{int(r['n_dongles_valid'])}),")
    return '\n'.join(out)

  # Per-dongle K spread at 120 km/h (speed_idx 5) for the caveat note.
  k_120 = per_dongle[(per_dongle['speed_idx'] == 5) & per_dongle['valid_passive']]['K'].to_numpy()
  std_120 = float(k_120.std()) if k_120.size >= 2 else float('nan')

  py = _PY_TEMPLATE.format(
    n_dongles=int(dongles.shape[0]),
    n_routes=int(routes.shape[0]),
    min_cc=5e-4,
    std_120=std_120,
    first_anchor=float(yaw['speed_kph'].iloc[0]),
    lateral_delay=float(dongles['lateral_delay_mean'].dropna().mean()),
    rows_yaw=_rows(yaw),
    rows_eps=_rows(eps),
    min_dongles=int(model_apply.MIN_DONGLES_VALID),
  )

  if out_py is None:
    out_py = run_dir / 'fleet_K_vw_id4_mk1.py'
  out_py.write_text(py)

  if out_json is None:
    out_json = run_dir / 'fleet_K_vw_id4_mk1.json'
  payload = {
    'fingerprint': 'VOLKSWAGEN_ID4_MK1',
    'n_dongles': int(dongles.shape[0]),
    'n_routes_ok': int(routes.shape[0]),
    'pooling': 'hierarchical (per-dongle then unweighted mean across dongles)',
    'min_dongles_valid_for_apply': int(model_apply.MIN_DONGLES_VALID),
    'yaw': yaw.to_dict(orient='records'),
    'eps': eps.to_dict(orient='records'),
  }
  out_json.write_text(json.dumps(payload, indent=2))
  return {'py': str(out_py), 'json': str(out_json),
          'n_dongles': int(dongles.shape[0]),
          'n_routes': int(routes.shape[0])}
