"""HTML + matplotlib report from a run directory."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openpilot.tools.lateral_fleet import aggregate, cache, features, hypotheses


def _save(fig, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=120, bbox_inches='tight')
  plt.close(fig)


def _heatmap(ax, mat: np.ndarray, title: str, vmax: float | None = None) -> None:
  with np.errstate(invalid='ignore'):
    v = float(np.nanmax(np.abs(mat))) if vmax is None else vmax
  if not np.isfinite(v) or v == 0:
    v = 1e-4
  im = ax.imshow(mat, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
  ax.set_title(title)
  ax.set_xlabel('|curvature| bucket')
  ax.set_ylabel('speed (km/h)')
  ax.set_yticks(range(features.NUM_SPEED_ANCHORS))
  ax.set_yticklabels([f'{int(s)}' for s in features.SPEED_ANCHORS_KPH])
  plt.colorbar(im, ax=ax, shrink=0.8)


def _residual_matrix(df: pd.DataFrame, col: str, sign: int) -> np.ndarray:
  mat = np.full((features.NUM_SPEED_ANCHORS, features.NUM_CURV_BUCKETS), np.nan)
  if df.empty:
    return mat
  sub = df[df['sign'] == sign]
  for _, row in sub.iterrows():
    si, ci = int(row['speed_idx']), int(row['curv_idx'])
    if 0 <= si < mat.shape[0] and 0 <= ci < mat.shape[1]:
      mat[si, ci] = row.get(col, np.nan)
  return mat


def plot_fleet_heatmaps(cross: pd.DataFrame, out_dir: Path) -> None:
  for kind in ('yaw', 'eps'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, sign in enumerate((1, -1)):
      _heatmap(axes[0, j], _residual_matrix(cross, f'resid_{kind}_mean_mean', sign),
               f'fleet mean residual ({kind}, sign={sign:+d})')
      _heatmap(axes[1, j], _residual_matrix(cross, f'gain_{kind}_mean_mean', sign),
               f'fleet mean K ({kind}, sign={sign:+d})')
    fig.suptitle(f'Fleet hierarchical: unweighted mean over dongles; (actual - c_cmd_delayed)',
                 y=1.02)
    _save(fig, out_dir / f'fleet_heatmap_{kind}.png')


def plot_per_dongle_K(pooled_by_dongle: dict[str, pd.DataFrame], out_dir: Path) -> None:
  for kind in ('yaw', 'eps'):
    gain_col = f'gain_{kind}_mean'
    fig, ax = plt.subplots(figsize=(9, 5))
    for d, p in sorted(pooled_by_dongle.items()):
      p = p[p['sufficient']]
      if p.empty:
        continue
      by_speed = p.groupby('speed_idx')[gain_col].mean()
      ax.plot([features.SPEED_ANCHORS_KPH[int(s)] for s in by_speed.index],
              by_speed.values, marker='o', alpha=0.7, label=d[:8])
    ax.axhline(1.0, color='black', lw=0.5, linestyle='--', label='K=1 ideal')
    ax.set_xlabel('speed (km/h)')
    ax.set_ylabel(f'mean K ({kind})')
    ax.set_title(f'Per-dongle steady-state EPS gain ({kind}-derived)')
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir / f'per_dongle_K_{kind}.png')


def plot_fleet_K(run_dir: Path) -> None:
  for kind in ('yaw', 'eps'):
    f = run_dir / f'plant_fleet_{kind}.parquet'
    if not f.exists():
      continue
    df = pd.read_parquet(f)
    fig, ax = plt.subplots(figsize=(9, 5))
    if df.empty:
      _save(fig, run_dir / f'plant_K_{kind}.png')
      continue
    ax.errorbar(df['speed_kph'], df['K_fleet_mean'],
                yerr=[df['K_fleet_mean'] - df['K_fleet_lo'],
                      df['K_fleet_hi'] - df['K_fleet_mean']],
                marker='o', capsize=4, label='fleet K (unweighted mean of dongles)')
    ax.axhline(1.0, color='black', lw=0.5, linestyle='--', label='K=1 (perfect tracking)')
    for _, r in df.iterrows():
      ax.annotate(f"n={int(r['n_dongles_valid'])}/{int(r['n_dongles_total'])}",
                  (r['speed_kph'], r['K_fleet_mean']), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('speed (km/h)')
    ax.set_ylabel(f'K ({kind})')
    ax.set_title(f'Parametric EPS plant gain K(v), {kind}-derived  (bootstrap CI over dongles)')
    ax.legend(fontsize=8)
    _save(fig, run_dir / f'plant_K_{kind}.png')


def plot_hypothesis_ranking(summary: pd.DataFrame, out_dir: Path) -> None:
  fig, ax = plt.subplots(figsize=(9, 5))
  for kind in ('yaw', 'eps'):
    sub = summary[summary['resid_kind'] == kind].sort_values('rms_after')
    x = np.arange(sub.shape[0])
    ax.plot(x, sub['rms_before'], 'o--', label=f'{kind} before', alpha=0.6)
    ax.plot(x, sub['rms_after'], 'o-', label=f'{kind} after')
    for i, name in enumerate(sub['hypothesis']):
      ax.annotate(name, (x[i], sub['rms_after'].iloc[i]), fontsize=7,
                  rotation=30, ha='left', va='bottom')
  ax.set_ylabel('RMS bucket residual  [rad/m]')
  ax.set_title('Hypothesis ranking')
  ax.legend(fontsize=8)
  _save(fig, out_dir / 'hypothesis_ranking.png')


_HTML_TPL = """<!doctype html>
<html><head><meta charset="utf-8"><title>lateral_fleet — EPS report</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; margin: 0.5em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 13px; }}
.warn {{ color: #b00; }}
.note {{ color: #555; font-style: italic; font-size: 12px; }}
</style></head><body>
<h1>lateral_fleet — EPS report</h1>
<p class="note">Run: {run_dir}. n_dongles={n_dongles}, n_routes={n_routes}.
Pooling: hierarchical (per-dongle first, then unweighted mean across dongles).</p>

<h2>Diagnostic-of-diagnostic</h2>
{diag_table}

<h2>Parametric EPS plant fit K(v)</h2>
<p class="note">Per-speed steady-state gain K with bootstrap CI across
dongles. K is the slope of a through-origin regression c_actual = K · c_cmd
on samples with |c_cmd| ≥ {min_cc:g} rad/m. K constrained to (0, 1] (passive
plant); dongles outside that range are dropped from the fleet mean.</p>
<img src="plant_K_yaw.png">
<img src="plant_K_eps.png">

<h2>Per-dongle K curves</h2>
<img src="per_dongle_K_yaw.png">
<img src="per_dongle_K_eps.png">

<h2>Fleet residual + K heatmaps</h2>
<img src="fleet_heatmap_yaw.png">
<img src="fleet_heatmap_eps.png">

<h2>Hypothesis ranking</h2>
{hyp_table}
<img src="hypothesis_ranking.png">

</body></html>
"""


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
  if df.empty:
    return '<p class="warn">(empty)</p>'
  return df.head(max_rows).to_html(index=False, float_format=lambda x: f'{x:.4g}')


def _quarantine_table(cache_root: Path = cache.CACHE_ROOT) -> str:
  counts: dict[str, int] = {}
  if not cache_root.exists():
    return '<p class="warn">no cache dir</p>'
  for dongle_dir in cache_root.iterdir():
    if not dongle_dir.is_dir():
      continue
    for status_file in dongle_dir.glob('*.status.json'):
      route_id = status_file.name[: -len('.status.json')]
      rs = cache.read_status(dongle_dir.name, route_id)
      if rs is None:
        counts['unreadable'] = counts.get('unreadable', 0) + 1
      else:
        counts[rs.status] = counts.get(rs.status, 0) + 1
  rows = '\n'.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>'
                   for k, v in sorted(counts.items()))
  return ('<table><thead><tr><th>status</th><th># routes</th></tr></thead>'
          f'<tbody>{rows}</tbody></table>')


def build_report(run_dir: Path,
                 cache_root: Path = cache.CACHE_ROOT,
                 min_dongles: int = aggregate.DEFAULT_MIN_DONGLES) -> Path:
  run_dir = Path(run_dir)
  routes = pd.read_parquet(run_dir / 'routes.parquet') if (run_dir / 'routes.parquet').exists() else pd.DataFrame()
  cross = pd.read_parquet(run_dir / 'cross_dongle.parquet') if (run_dir / 'cross_dongle.parquet').exists() else pd.DataFrame()
  dongle_meta = pd.read_parquet(run_dir / 'dongles.parquet') if (run_dir / 'dongles.parquet').exists() else pd.DataFrame()
  pooled_dir = run_dir / 'dongle_pooled'
  pooled_by_dongle: dict[str, pd.DataFrame] = {}
  if pooled_dir.exists():
    for f in pooled_dir.glob('*.parquet'):
      pooled_by_dongle[f.stem] = pd.read_parquet(f)

  plot_fleet_heatmaps(cross, run_dir)
  plot_per_dongle_K(pooled_by_dongle, run_dir)
  plot_fleet_K(run_dir)
  summary = hypotheses.evaluate_all(pooled_by_dongle, dongle_meta)
  summary.to_parquet(run_dir / 'hypothesis_summary.parquet', index=False)
  plot_hypothesis_ranking(summary, run_dir)

  diag_table = _quarantine_table(cache_root)
  hyp_table = _df_to_html(summary)

  from openpilot.tools.lateral_fleet import plant_fit
  out = _HTML_TPL.format(
    run_dir=html.escape(str(run_dir)),
    n_dongles=len(pooled_by_dongle),
    n_routes=int(routes.shape[0]),
    min_dongles=min_dongles,
    diag_table=diag_table,
    hyp_table=hyp_table,
    min_cc=plant_fit.MIN_C_CMD_FIT,
  )
  out_path = run_dir / 'report.html'
  out_path.write_text(out)
  return out_path
