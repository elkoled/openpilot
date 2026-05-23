#!/usr/bin/env python3
"""
Stage 5 — Render fleet_stats.pkl to a single HTML report.

Follows the matplotlib-base64-embed pattern used by
tools/longitudinal_maneuvers/generate_report.py. Sections:

  1. Executive summary + N_cars / N_segments / total engaged hours.
  2. Population bucket heatmaps (learner-gated + extended-ungated).
  3. Inside-vs-outside grid mass distribution.
  4. Hypothesis battery: per-car gain, lag, asymmetry, residual percentiles.
  5. Covariate Spearman correlations (with N).
  6. Learner-replay vs offline-batch-fit agreement per car.
  7. PID-on vs PID-off engaged residual comparison.
  8. Decision tree application — applied here, not in analyze.py.
  9. Per-car appendix (one bucket heatmap per dongle, only for cars with
     >= MIN_CARS_PER_PLOT engaged hours).

Buckets with fewer than MIN_CARS_PER_BUCKET contributing cars are greyed out.
Plots with fewer than MIN_CARS_PER_PLOT total cars are hidden with a
"insufficient cars" placeholder.
"""
from __future__ import annotations

import argparse
import base64
import io
import pickle
import sys
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _img_tag(fig, max_width: int = 1000) -> str:
  buf = io.BytesIO()
  fig.savefig(buf, format="webp", bbox_inches="tight", dpi=120)
  plt.close(fig)
  b64 = base64.b64encode(buf.getvalue()).decode()
  return f"<img src='data:image/webp;base64,{b64}' style='width:100%; max-width:{max_width}px;'>"


def _bucket_heatmap(median: np.ndarray, n_cars: np.ndarray, title: str,
                    speed_anchors_kph, curvature_edges, min_cars_per_bucket: int,
                    vsymmetric=True) -> str:
  S, C = median.shape
  m = median.copy()
  # grey out buckets with insufficient cars (force to NaN so cmap masks them)
  m[n_cars < min_cars_per_bucket] = np.nan
  fig, ax = plt.subplots(figsize=(12, 5))
  if vsymmetric:
    abs_max = float(np.nanmax(np.abs(m))) if np.any(np.isfinite(m)) else 1e-4
    vmin, vmax = -abs_max, abs_max
    cmap = "RdBu_r"
  else:
    vmin, vmax = None, None
    cmap = "viridis"
  im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
  ax.set_yticks(range(S))
  ax.set_yticklabels([f"{kph:.0f}" for kph in speed_anchors_kph])
  ax.set_ylabel("v_ego (km/h)")
  centers = np.sqrt(curvature_edges[:-1] * curvature_edges[1:])
  ax.set_xticks(range(C))
  ax.set_xticklabels([f"{c:.0e}" for c in centers], rotation=60)
  ax.set_xlabel("|curvature| (rad/m)")
  ax.set_title(title)
  for s in range(S):
    for c in range(C):
      n = int(n_cars[s, c])
      if n >= 1:
        color = "k" if (np.isfinite(m[s, c]) and abs(m[s, c]) < (abs_max * 0.5)) else "w"
        ax.text(c, s, str(n), ha="center", va="center", color=color, fontsize=7)
  cb = fig.colorbar(im, ax=ax)
  cb.set_label("median signed residual (rad/m)")
  return _img_tag(fig)


def _scatter_with_rho(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                      title: str, rho: float, n: int) -> str:
  fig, ax = plt.subplots(figsize=(6, 5))
  m = np.isfinite(x) & np.isfinite(y)
  if int(np.sum(m)) >= 3:
    ax.scatter(x[m], y[m], alpha=0.7)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}  (N={n})")
  ax.grid(True, alpha=0.3)
  return _img_tag(fig, max_width=600)


def _decision_summary(stats: dict) -> str:
  """Apply the pre-registered decision tree from the plan to the data."""
  pop = stats["pop_learner_ungated"]
  median = np.array(pop["median"], dtype=np.float64)
  n_cars = np.array(pop["n_cars"], dtype=np.int64)
  total_mass = float(np.nansum(np.abs(median)))
  # inside-mass fractions
  inside = np.array([v for v in stats["inside_mass_per_car"].values()
                     if v is not None and np.isfinite(v)], dtype=np.float64)
  inside_median = float(np.nanmedian(inside)) if len(inside) else float("nan")
  covar = stats["covariate_spearman"]
  max_rho = max((abs(v["rho"]) for v in covar.values() if np.isfinite(v["rho"])), default=0.0)
  lr_vs_bf = np.array(stats["hypothesis_battery"]["learner_vs_batch_pearson"], dtype=np.float64)
  lr_vs_bf = lr_vs_bf[np.isfinite(lr_vs_bf)]
  lr_vs_bf_median = float(np.median(lr_vs_bf)) if len(lr_vs_bf) else float("nan")

  decisions = []
  decisions.append(f"Median fraction of residual mass inside learner-supported region: {inside_median:.2f}")
  decisions.append(f"Strongest covariate |ρ|: {max_rho:.2f}")
  decisions.append(f"Learner-replay vs offline-batch-fit Pearson median across cars: {lr_vs_bf_median:.2f}")

  rec = []
  if not np.isfinite(inside_median):
    rec.append("INSUFFICIENT DATA — no per-car inside-fraction available. Ship nothing; gather more.")
  elif inside_median < 0.5:
    rec.append("Residual mass concentrates OUTSIDE the learner-supported region. "
               "The dynamic_steering learner cannot help by construction. "
               "Recommend extending the bucket grid or a different mechanism.")
  elif lr_vs_bf_median > 0.6 and inside_median > 0.5:
    rec.append("Learner converges to the offline batch-fit answer (Pearson > 0.6) "
               "AND residual lives inside the supported region. Defensible to ship "
               "the learner-as-is, with the cross-car variance acknowledged.")
  elif max_rho < 0.2:
    rec.append("No single observable explains per-car residual variation (|ρ| < 0.2 "
               "for every candidate). Likely a per-car calibration issue. Recommend "
               "forcing liveParameters re-learning or VIN-keyed reset; do not ship a "
               "global controller patch.")
  else:
    rec.append("Mixed signal: residual is inside the grid but the learner doesn't "
               "converge to the batch-fit answer, OR one covariate explains a "
               "moderate fraction (|ρ| ≥ 0.2). Investigate the dominant covariate "
               "before shipping anything.")

  return "<ul>" + "".join(f"<li>{escape(d)}</li>" for d in decisions) + \
         "</ul><div class='rec'><b>Recommendation:</b><br>" + \
         "<br>".join(escape(r) for r in rec) + "</div>"


def render(stats: dict, out_path: str) -> None:
  ANCHORS_KPH = (20, 40, 60, 80, 100, 120, 140)
  EXT_ANCHORS_KPH = (20, 40, 60, 80, 100, 120, 140, 150, 160)
  LEARNER_EDGES = np.array([1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5,
                            1.28e-4, 2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3])
  EXT_EDGES = np.concatenate([LEARNER_EDGES, np.array([8.192e-3, 1.6384e-2, 3.2768e-2, 6.5536e-2])])

  parts = []
  parts.append("""<!doctype html>
<html><head><meta charset='utf-8'><title>ID4_MK1 Lateral Tracking Investigation</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { border-bottom: 2px solid #555; }
  h2 { margin-top: 2em; border-bottom: 1px solid #aaa; }
  .meta { color: #555; font-size: 0.9em; }
  .warn { background: #fff7d6; padding: 0.5em 1em; border-left: 4px solid #cc9900; }
  .rec  { background: #e8f4ff; padding: 0.5em 1em; border-left: 4px solid #1f6feb; margin: 1em 0; }
  table { border-collapse: collapse; }
  td, th { padding: 4px 10px; border: 1px solid #ccc; font-size: 0.9em; }
  th { background: #f3f3f3; }
</style>
</head><body>
""")
  parts.append("<h1>VOLKSWAGEN_ID4_MK1 — Lateral Tracking Investigation</h1>")
  parts.append(f"<div class='meta'>N_cars = {stats['n_cars']}, "
               f"N_segments = {stats['n_segments']}, "
               f"min_cars_per_bucket = {stats['min_cars_per_bucket']}</div>")

  # Section 1: executive summary
  parts.append("<h2>Decision summary</h2>")
  parts.append(_decision_summary(stats))

  if stats["n_cars"] < stats["min_cars_per_plot"]:
    parts.append(f"<div class='warn'>FLEET-LEVEL N WARNING: only {stats['n_cars']} "
                 f"cars contributed. Population-level conclusions below are "
                 "not defensible. Skipping fleet figures.</div>")
    parts.append("</body></html>")
    with open(out_path, "w") as f:
      f.write("".join(parts))
    return

  # Section 2: population bucket heatmaps
  parts.append("<h2>Population residual on learner-supported grid (gated)</h2>")
  m = np.array(stats["pop_learner_gated"]["median"], dtype=np.float64)
  n = np.array(stats["pop_learner_gated"]["n_cars"], dtype=np.int64)
  parts.append(_bucket_heatmap(m, n, "Median per-car signed residual (gated)",
                               ANCHORS_KPH, LEARNER_EDGES, stats["min_cars_per_bucket"]))

  parts.append("<h2>Population residual on extended grid (ungated, engaged only)</h2>")
  parts.append("<div class='meta'>Cells with fewer than "
               f"{stats['min_cars_per_bucket']} contributing cars are blanked. "
               "Numbers in each cell are N_cars.</div>")
  m = np.array(stats["pop_extended_ungated"]["median"], dtype=np.float64)
  n = np.array(stats["pop_extended_ungated"]["n_cars"], dtype=np.int64)
  parts.append(_bucket_heatmap(m, n, "Median per-car signed residual (ungated, extended)",
                               EXT_ANCHORS_KPH, EXT_EDGES, stats["min_cars_per_bucket"]))

  # Section 3: inside-vs-outside mass
  parts.append("<h2>Residual mass inside learner-supported region</h2>")
  inside = np.array([v for v in stats["inside_mass_per_car"].values()
                     if v is not None and np.isfinite(v)], dtype=np.float64)
  if len(inside) >= stats["min_cars_per_plot"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inside, bins=20, range=(0, 1), edgecolor="k")
    ax.set_xlabel("fraction of |signed residual mass| inside learner region")
    ax.set_ylabel("# cars")
    ax.set_title(f"Inside-grid mass fraction, N_cars = {len(inside)}")
    ax.axvline(np.median(inside), color="r", linestyle="--", label=f"median = {np.median(inside):.2f}")
    ax.legend()
    parts.append(_img_tag(fig, max_width=700))
  else:
    parts.append(f"<div class='warn'>N_cars = {len(inside)} &lt; {stats['min_cars_per_plot']}, plot hidden.</div>")

  # Section 4: hypothesis battery scatter plots vs highway_p95
  parts.append("<h2>Hypothesis battery — covariates vs per-car residual (highway p95)</h2>")
  hb = stats["hypothesis_battery"]
  y = np.array(hb["highway_p95"], dtype=np.float64)
  for cov_name in ("steer_ratio", "stiffness", "lateral_delay", "eps_power",
                   "engaged_s", "gain_80kph", "gain_100kph", "gain_120kph"):
    x = np.array(hb.get(cov_name, []), dtype=np.float64)
    cov_entry = stats["covariate_spearman"].get(cov_name, {"rho": float("nan"), "n": 0})
    if len(x) != len(y) or cov_entry["n"] < stats["min_cars_per_plot"]:
      parts.append(f"<div class='warn'>{escape(cov_name)}: N = {cov_entry['n']} "
                   f"&lt; {stats['min_cars_per_plot']}, skipping.</div>")
      continue
    parts.append(_scatter_with_rho(x, y, cov_name, "highway |residual| p95 (rad/m)",
                                   f"{cov_name} vs highway residual",
                                   cov_entry["rho"], cov_entry["n"]))

  # Section 5: learner-replay vs batch-fit
  parts.append("<h2>Learner replay vs offline batch fit (per car Pearson)</h2>")
  lr_vs_bf = np.array(hb["learner_vs_batch_pearson"], dtype=np.float64)
  lr_vs_bf_f = lr_vs_bf[np.isfinite(lr_vs_bf)]
  if len(lr_vs_bf_f) >= stats["min_cars_per_plot"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lr_vs_bf_f, bins=20, range=(-1, 1), edgecolor="k")
    ax.set_xlabel("Pearson ρ between online learner bias and offline batch fit")
    ax.set_ylabel("# cars")
    ax.set_title(f"Learner convergence quality, N_cars = {len(lr_vs_bf_f)}")
    ax.axvline(np.median(lr_vs_bf_f), color="r", linestyle="--",
               label=f"median = {np.median(lr_vs_bf_f):.2f}")
    ax.legend()
    parts.append(_img_tag(fig, max_width=700))
  else:
    parts.append(f"<div class='warn'>N_cars = {len(lr_vs_bf_f)} &lt; "
                 f"{stats['min_cars_per_plot']}, plot hidden.</div>")

  # Section 6: PID vs no-PID
  parts.append("<h2>PID-on vs PID-off engaged residual (highway p95)</h2>")
  pid_cmp = stats["pid_vs_nopid"]
  parts.append(f"<table><tr><th>cohort</th><th>N_cars</th><th>median</th><th>90% CI</th></tr>"
               f"<tr><td>PID on</td><td>{pid_cmp['pid_on']['n_cars']}</td>"
               f"<td>{pid_cmp['pid_on']['median']:.5f}</td>"
               f"<td>[{pid_cmp['pid_on']['ci_lo']:.5f}, {pid_cmp['pid_on']['ci_hi']:.5f}]</td></tr>"
               f"<tr><td>PID off</td><td>{pid_cmp['pid_off']['n_cars']}</td>"
               f"<td>{pid_cmp['pid_off']['median']:.5f}</td>"
               f"<td>[{pid_cmp['pid_off']['ci_lo']:.5f}, {pid_cmp['pid_off']['ci_hi']:.5f}]</td></tr></table>")
  if min(pid_cmp['pid_on']['n_cars'], pid_cmp['pid_off']['n_cars']) < stats["min_cars_per_plot"]:
    parts.append(f"<div class='warn'>One cohort has &lt; {stats['min_cars_per_plot']} cars; "
                 "cohort comparison underpowered.</div>")

  # Section 7: per-car appendix (compact summary table)
  parts.append("<h2>Per-car summary</h2>")
  cars_summary = sorted(stats["cars"], key=lambda c: c.get("highway_p95_median") or 1e9)
  parts.append("<table><tr><th>dongle</th><th>segments</th><th>ok</th>"
               "<th>engaged (min)</th><th>VIN</th><th>build</th>"
               "<th>PID?</th><th>p50</th><th>p95</th></tr>")
  for c in cars_summary:
    p50 = c.get("highway_p50_median")
    p95 = c.get("highway_p95_median")
    parts.append(
      f"<tr><td>{escape(c['dongle'])}</td>"
      f"<td>{c['segments']}</td><td>{c['ok_segments']}</td>"
      f"<td>{c['engaged_s']/60:.1f}</td>"
      f"<td>{escape(c.get('vin','')[:17])}</td>"
      f"<td>{escape(c.get('build_year','') or '')}</td>"
      f"<td>{'Y' if c['has_pid'] else 'N'}</td>"
      f"<td>{p50:.5f}</td><td>{p95:.5f}</td></tr>"
    )
  parts.append("</table>")

  parts.append("</body></html>")
  with open(out_path, "w") as f:
    f.write("".join(parts))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--stats", default="fleet_stats.pkl")
  p.add_argument("--out", default="fleet_report.html")
  args = p.parse_args()
  with open(args.stats, "rb") as f:
    stats = pickle.load(f)
  render(stats, args.out)
  print(f"[report] wrote {args.out}", flush=True)


if __name__ == "__main__":
  sys.exit(main())
