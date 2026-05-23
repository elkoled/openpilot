"""Minimal HTML report writer. Style follows tools/lateral_maneuvers/generate_report.py:
matplotlib for plots, base64-embedded PNGs, string-builder HTML.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from .aggregate import DongleSummary
from .features import SPEED_ANCHORS, CURVATURE_BUCKET_EDGES, LAG_GRID_S
from .hypotheses import CURVATURE_BUCKET_CENTERS

PAIR_LABELS = ("P1_desired_yaw", "P2_desired_qfk", "P3_apply_yaw", "P4_apply_qfk")


def _png_b64(fig) -> str:
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
  import matplotlib.pyplot as plt
  plt.close(fig)
  return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_bucket_heatmap(features: dict, label: str, title: str):
  import matplotlib.pyplot as plt
  c = features[f"{label}_count"]; s = features[f"{label}_sum_residual"]
  with np.errstate(invalid="ignore", divide="ignore"):
    mean = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
  fig, ax = plt.subplots(figsize=(7, 3.5))
  im = ax.imshow(mean, aspect="auto", cmap="RdBu_r",
                 vmin=-np.nanmax(np.abs(mean)) if np.any(np.isfinite(mean)) else -1,
                 vmax=+np.nanmax(np.abs(mean)) if np.any(np.isfinite(mean)) else +1)
  ax.set_yticks(range(7))
  ax.set_yticklabels([f"{int(v*3.6)}kph" for v in SPEED_ANCHORS])
  ax.set_xticks(range(len(CURVATURE_BUCKET_CENTERS)))
  ax.set_xticklabels([f"{c:.0e}" for c in CURVATURE_BUCKET_CENTERS], rotation=60, ha="right", fontsize=7)
  ax.set_xlabel("|desired curvature| (rad/m)")
  ax.set_title(title)
  fig.colorbar(im, ax=ax, label="signed mean residual (rad/m)")
  return _png_b64(fig)


def _plot_xcorr(features: dict, label: str, title: str):
  import matplotlib.pyplot as plt
  r = features[f"{label}_xcorr_r"]
  fig, ax = plt.subplots(figsize=(6, 2.5))
  ax.plot(LAG_GRID_S * 1000, r, marker="o")
  ax.axvline(0, color="k", alpha=0.3, linewidth=0.5)
  ax.set_xlabel("desired-leads-actual lag (ms)")
  ax.set_ylabel("Pearson r")
  ax.set_title(title)
  ax.grid(True, alpha=0.3)
  return _png_b64(fig)


def _plot_fleet_distribution(dongles: list[DongleSummary]):
  import matplotlib.pyplot as plt
  scores = np.array([d.scalar_tracking_score for d in dongles
                     if np.isfinite(d.scalar_tracking_score)])
  if scores.size == 0:
    return ""
  fig, ax = plt.subplots(figsize=(7, 3.0))
  ax.hist(scores, bins=max(5, min(20, scores.size)), edgecolor="black")
  ax.set_xlabel("scalar tracking score (RMS rad/m, lower=better)")
  ax.set_ylabel("dongles")
  ax.set_title(f"Cross-dongle tracking score distribution (N={scores.size})")
  return _png_b64(fig)


def write_html_report(dongles: list[DongleSummary], decision: dict, out_path: Path) -> None:
  parts = ["<!doctype html><html><head><meta charset='utf-8'>"
           "<title>ID4 MK1 fleet lateral tracking</title>"
           "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
           "h1,h2,h3{margin-top:1.5em}img{max-width:100%}table{border-collapse:collapse}"
           "th,td{border:1px solid #ccc;padding:4px 8px;font-size:90%}.warn{color:#a33}"
           "details{margin:0.5em 0}</style></head><body>"]
  parts.append("<h1>ID4 MK1 fleet lateral tracking</h1>")
  parts.append(f"<p><b>Recommendation:</b> <code>{decision['recommendation']}</code></p>")
  if "rationale" in decision:
    parts.append(f"<p>{decision['rationale']}</p>")
  if decision.get("warnings"):
    parts.append("<div class='warn'><b>Warnings:</b><ul>")
    for w in decision["warnings"]:
      parts.append(f"<li>{w}</li>")
    parts.append("</ul></div>")
  parts.append("<h2>Per-hypothesis winners</h2><ul>")
  for h, n in sorted(decision.get("per_hypothesis_count", {}).items(), key=lambda kv: -kv[1]):
    parts.append(f"<li><code>{h}</code>: {n}</li>")
  parts.append("</ul>")

  fleet_img = _plot_fleet_distribution(dongles)
  if fleet_img:
    parts.append(f"<h2>Cross-dongle distribution</h2><img src='data:image/png;base64,{fleet_img}'/>")

  parts.append("<h2>Per-dongle leaderboard</h2><table><thead><tr>"
               "<th>dongle</th><th>routes</th><th>engaged_s</th><th>gated_n</th>"
               "<th>tracking score</th><th>winner</th><th>VIN</th></tr></thead><tbody>")
  for d in dongles[:50]:
    parts.append(
      f"<tr><td>{d.dongle_id}</td><td>{d.n_routes}</td>"
      f"<td>{d.engaged_seconds:.0f}</td><td>{d.gated_samples}</td>"
      f"<td>{d.scalar_tracking_score:.3e}</td>"
      f"<td><code>{d.hypothesis['winner']}</code></td><td>{d.vin}</td></tr>"
    )
  parts.append("</tbody></table>")

  for d in dongles[:10]:
    parts.append(f"<h3>{d.dongle_id} &mdash; routes={d.n_routes}, VIN {d.vin}</h3>")
    parts.append("<details><summary>Bucket residual heatmaps + xcorr</summary>")
    for label in PAIR_LABELS:
      title = f"{d.dongle_id} {label}"
      img = _plot_bucket_heatmap(d.pooled_features, label, title)
      parts.append(f"<img src='data:image/png;base64,{img}'/>")
      img_x = _plot_xcorr(d.pooled_features, label, f"{title} (xcorr)")
      parts.append(f"<img src='data:image/png;base64,{img_x}'/>")
    parts.append("</details>")

  parts.append("</body></html>")
  out_path.write_text("".join(parts))
