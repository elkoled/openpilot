"""Loads per-route residual pickles and produces a text + HTML report.

Single-route: descriptive view of the five residual surfaces, lag table,
deadband/asymmetry, metadata, plus heatmaps.

Multi-route: side-by-side text table and per-residual fleet histograms.
Cross-fleet aggregate statistics (population means, mixed-effects fits,
cluster-bootstrap CIs) intentionally require N >= a sensible minimum
(default 10 dongles) and emit a clear "insufficient population" message
otherwise.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import math
import os
import pickle
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# Late import of matplotlib so --help is fast.

RESIDUAL_NAMES = ("R_model_yaw", "R_model_qfk", "R_hca_qfk", "R_hca_yaw", "R_smooth_loss")
RESIDUAL_LABELS = {
  'R_model_yaw': 'model_raw  -  yaw_actual    (learner view, outer envelope)',
  'R_model_qfk': 'model_raw  -  QFK_01 rack   (model vs rack-measured)',
  'R_hca_qfk':   'HCA_03 cmd -  QFK_01 rack   (EPS execution)',
  'R_hca_yaw':   'HCA_03 cmd -  yaw_actual    (EPS -> plant)',
  'R_smooth_loss':'model_raw -  actuators_curv (controller smoothing)',
}


def load_summary(path: str) -> dict:
  with open(path, 'rb') as f:
    return pickle.load(f)


def per_residual_stats(s: dict, name: str) -> dict:
  r = s['residuals'][name]
  c = r['count']
  total = int(c.sum())
  if total == 0:
    return {'n': 0}
  return {
    'n': total,
    'mean': float(r['sum'].sum() / total),
    'rms': float(math.sqrt(r['sum_sq'].sum() / total)),
    'mae': float(r['sum_abs'].sum() / total),
  }


def fmt_eps(v: float | None, digits: int = 2) -> str:
  if v is None:
    return ' n/a '
  return f"{v:+.{digits}e}"


def text_report_single(s: dict) -> str:
  out = []
  out.append(f"route_id          : {s.get('route_id')}")
  out.append(f"fingerprint       : {s.get('fingerprint')}    brand={s.get('brand')}")
  out.append(f"vin               : {s.get('vin') or '(missing)'}")
  out.append(f"branch_inferred   : {s.get('branch_inferred')}")
  out.append(f"flags             : {s.get('flags')}    steer_control_type={s.get('steer_control_type')}")
  out.append(f"has_can           : {s.get('has_can')}   buses(hca,qfk)={s.get('can_bus_used')}")
  if 'error' in s:
    out.append(f"ERROR             : {s['error']}")
    return "\n".join(out)
  out.append(f"duration_s        : {s.get('duration_s'):.1f}    grid_samples={s.get('grid_samples')}")
  out.append(f"apply_samples     : {s.get('apply_samples')}")
  out.append(f"learn_samples     : {s.get('learn_samples')}")
  out.append(f"lat_delay_used_s  : {s.get('lat_delay_used_s'):.3f}")
  sp = s.get('speed_kmh', {})
  if sp.get('p50') is not None:
    out.append(f"speed kmh p10/50/90/99 : {sp['p10']:.1f} / {sp['p50']:.1f} / {sp['p90']:.1f} / {sp['p99']:.1f}")
  lsr = s.get('lp_steer_ratio', {})
  out.append(f"steer_ratio       : mean={lsr.get('mean')} std={lsr.get('std')}  n={lsr.get('n')}")
  lst = s.get('lp_stiffness', {})
  out.append(f"stiffness_factor  : mean={lst.get('mean')} std={lst.get('std')}")
  lao = s.get('lp_angle_offset', {})
  out.append(f"angle_offset_deg  : mean={lao.get('mean')} std={lao.get('std')}")
  lld = s.get('ld_lat_delay', {})
  out.append(f"liveDelay         : mean={lld.get('mean')} std={lld.get('std')}")
  out.append("")
  out.append(f"{'residual':<14}{'n':>8}{'mean':>14}{'rms':>14}{'mae':>14}  description")
  out.append("-" * 110)
  for name in RESIDUAL_NAMES:
    st = per_residual_stats(s, name)
    if st['n'] == 0:
      out.append(f"{name:<14}{0:>8}  (no samples)")
      continue
    out.append(f"{name:<14}{st['n']:>8}{fmt_eps(st['mean']):>14}{fmt_eps(st['rms']):>14}{fmt_eps(st['mae']):>14}  {RESIDUAL_LABELS[name]}")

  out.append("")
  out.append("Lag (sec) - cross-correlation of d/dt(model_raw_desired) vs d/dt(yaw_actual).")
  out.append("Pre-shifted by liveDelay; reported value is the residual lag.")
  for band, info in s.get('lags', {}).items():
    if info.get('lag_s') is None:
      out.append(f"  {band:14s}: insufficient samples (n={info.get('samples', 0)})")
    else:
      out.append(f"  {band:14s}: {info['lag_s']:+.3f} s  (n={info['samples']}, peak xcorr {info.get('peak_xcorr', 0):.2f}, zero-lag xcorr {info.get('zero_xcorr', 0):.2f})")
  out.append("")
  db = s.get('deadband', {})
  out.append("Deadband signature (R_model_yaw signed bias in smallest 4 |curv| buckets):")
  for k, v in db.items():
    out.append(f"  {k}: n={v.get('count')}  mean_signed_bias={fmt_eps(v.get('mean_signed_bias'))}")
  out.append("")
  out.append("Asymmetry signature (pos+neg signed-bias sum per speed; nonzero => left/right mismatch):")
  for k, v in s.get('asymmetry', {}).items():
    sum_v = v.get('sum')
    out.append(f"  {k:8s}: pos_mean={fmt_eps(v.get('pos_mean'))}  neg_mean={fmt_eps(v.get('neg_mean'))}  sum={fmt_eps(sum_v)}  n+={v.get('pos_count')} n-={v.get('neg_count')}")
  return "\n".join(out)


# ---------- plotting ----------

def _png_b64(fig) -> str:
  import matplotlib.pyplot as plt
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
  plt.close(fig)
  buf.seek(0)
  return base64.b64encode(buf.read()).decode('ascii')


def plot_residual_heatmap(s: dict, name: str, sign_idx: int = None) -> str:
  """Returns HTML <img> tag (base64 PNG) for the residual mean heatmap.
  If sign_idx is None, sum across signs (symmetric magnitude view).
  """
  import matplotlib.pyplot as plt
  r = s['residuals'][name]
  c = r['count'].copy()
  m = r['sum'].copy()
  abs_m = r['sum_abs'].copy()
  if sign_idx is None:
    c_total = c.sum(axis=2)
    m_total = m.sum(axis=2)
    abs_total = abs_m.sum(axis=2)
  else:
    c_total = c[:, :, sign_idx]
    m_total = m[:, :, sign_idx]
    abs_total = abs_m[:, :, sign_idx]
  with np.errstate(invalid='ignore', divide='ignore'):
    mean_grid = np.where(c_total > 0, m_total / np.maximum(c_total, 1), np.nan)
    mae_grid = np.where(c_total > 0, abs_total / np.maximum(c_total, 1), np.nan)

  speeds = s['bucket_grid']['speed_anchors_kmh']
  curvs = s['bucket_grid']['curv_centers']

  fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)

  for ax, grid, title, cmap, vmag in (
      (axes[0], np.log10(np.maximum(c_total, 1)), 'log10 count', 'viridis', None),
      (axes[1], mean_grid,                          'mean signed bias', 'RdBu_r', None),
      (axes[2], mae_grid,                           'mean abs error',  'magma',  None),
  ):
    if vmag is None and title == 'mean signed bias':
      finite = mean_grid[np.isfinite(mean_grid)]
      if finite.size:
        vmag = float(np.max(np.abs(finite)))
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap=cmap,
                   vmin=(-vmag if vmag else None), vmax=(vmag if vmag else None))
    ax.set_title(title)
    ax.set_xticks(range(len(curvs)))
    ax.set_xticklabels([f"{c:.0e}" for c in curvs], rotation=60, fontsize=7)
    ax.set_yticks(range(len(speeds)))
    ax.set_yticklabels([f"{int(v)}" for v in speeds])
    ax.set_xlabel("|curvature| bucket center (1/m)")
    if ax is axes[0]:
      ax.set_ylabel("speed (km/h)")
    plt.colorbar(im, ax=ax, fraction=0.046)
  sign_label = 'both' if sign_idx is None else ('positive' if sign_idx == 1 else 'negative')
  fig.suptitle(f"{name}  -  {RESIDUAL_LABELS[name]}  (sign: {sign_label})", fontsize=10)
  fig.tight_layout()
  b64 = _png_b64(fig)
  return f'<img src="data:image/png;base64,{b64}" />'


def html_report_single(s: dict, out_path: str) -> str:
  parts: list[str] = []
  parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Fleet residual report</title>")
  parts.append("<style>body{font-family:monospace} pre{white-space:pre-wrap} h2{margin-top:30px} table{border-collapse:collapse} td,th{padding:4px 8px;border:1px solid #999}</style>")
  parts.append("</head><body>")
  parts.append(f"<h1>Lateral residual report</h1>")
  parts.append(f"<h3>{s.get('route_id')}</h3>")
  parts.append(f"<pre>{text_report_single(s)}</pre>")
  if 'residuals' in s and s.get('learn_samples', 0) > 0:
    parts.append("<h2>Residual surfaces</h2>")
    parts.append("<p>Three panels per residual: log10 count, mean signed bias, mean absolute error. Speeds on y, |curvature| bucket centers on x. 'sign: both' aggregates left+right; per-sign panels follow.</p>")
    for name in RESIDUAL_NAMES:
      if per_residual_stats(s, name)['n'] == 0:
        continue
      parts.append(f"<h3>{name}</h3>")
      parts.append(plot_residual_heatmap(s, name, sign_idx=None))
      parts.append(plot_residual_heatmap(s, name, sign_idx=1))
      parts.append(plot_residual_heatmap(s, name, sign_idx=0))
  parts.append("</body></html>")
  html = "\n".join(parts)
  with open(out_path, 'w') as f:
    f.write(html)
  return out_path


def text_report_multi(summaries: list[dict]) -> str:
  out = []
  out.append(f"Loaded {len(summaries)} route summary(ies).")
  out.append("")
  out.append(f"{'route':40s}{'fingerprint':20s}{'branch':24s}{'samples':>10s}{'R_model_yaw RMS':>18s}{'R_hca_qfk RMS':>16s}")
  out.append("-" * 130)
  for s in summaries:
    rid = s.get('route_id', '?')[:38]
    fp = (s.get('fingerprint', '') or '')[:18]
    br = (s.get('branch_inferred', '') or '')[:22]
    ns = s.get('learn_samples', 0) or 0
    rms_my = per_residual_stats(s, 'R_model_yaw').get('rms')
    rms_hq = per_residual_stats(s, 'R_hca_qfk').get('rms')
    out.append(f"{rid:40s}{fp:20s}{br:24s}{ns:>10}{fmt_eps(rms_my):>18s}{fmt_eps(rms_hq):>16s}")
  out.append("")
  # Highlight where in the chain the residual lives (median across routes).
  out.append("Where does the residual concentrate (median RMS across routes):")
  for name in RESIDUAL_NAMES:
    vals = []
    for s in summaries:
      st = per_residual_stats(s, name)
      if st.get('n', 0) > 100:
        vals.append(st['rms'])
    if not vals:
      out.append(f"  {name:14s}: no routes with enough samples")
      continue
    med = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    out.append(f"  {name:14s}: median RMS {fmt_eps(med)}   p10 {fmt_eps(p10)}   p90 {fmt_eps(p90)}   (n_routes={len(vals)})")
  out.append("")
  n_dongles = len({(s.get('route_id', '/').split('/')[0]) for s in summaries})
  out.append(f"Dongle count: {n_dongles}")
  if n_dongles < 10:
    out.append("INSUFFICIENT POPULATION: cross-dongle stats need >=10 unique dongles for cluster-bootstrap CIs to be meaningful. Treat the table above as indicative only.")
  return "\n".join(out)


def html_report_multi(summaries: list[dict], out_path: str) -> str:
  parts: list[str] = []
  parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Fleet residual report</title>")
  parts.append("<style>body{font-family:monospace} pre{white-space:pre-wrap}</style></head><body>")
  parts.append("<h1>Fleet residual report</h1>")
  parts.append(f"<pre>{text_report_multi(summaries)}</pre>")
  parts.append("<h2>Per-route detail</h2>")
  for s in summaries:
    parts.append(f"<h3>{s.get('route_id')}</h3>")
    parts.append(f"<pre>{text_report_single(s)}</pre>")
  parts.append("</body></html>")
  html = "\n".join(parts)
  with open(out_path, 'w') as f:
    f.write(html)
  return out_path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('paths', nargs='+',
                  help='One or more pickle files or globs. Use tools/lateral_maneuvers/fleet_residuals/out/*.pkl after a fleet run.')
  ap.add_argument('--html-out', default=None,
                  help='Path to write an HTML report. Defaults to tools/lateral_maneuvers/fleet_residuals/out/report.html')
  ap.add_argument('--no-html', action='store_true')
  args = ap.parse_args()

  paths = []
  for p in args.paths:
    if any(c in p for c in '*?['):
      paths.extend(sorted(glob.glob(p)))
    else:
      paths.append(p)
  if not paths:
    print("no pickles found", file=sys.stderr)
    sys.exit(1)

  summaries = []
  for p in paths:
    try:
      summaries.append(load_summary(p))
    except Exception as e:
      print(f"  skip {p}: {e}", file=sys.stderr)

  if not summaries:
    print("no summaries loaded", file=sys.stderr)
    sys.exit(1)

  if len(summaries) == 1:
    print(text_report_single(summaries[0]))
    if not args.no_html:
      out = args.html_out or os.path.join(os.path.dirname(paths[0]), 'report.html')
      html_report_single(summaries[0], out)
      print(f"\nHTML report: {out}")
  else:
    print(text_report_multi(summaries))
    if not args.no_html:
      out = args.html_out or os.path.join(os.path.dirname(paths[0]), 'report.html')
      html_report_multi(summaries, out)
      print(f"\nHTML report: {out}")


if __name__ == '__main__':
  main()
