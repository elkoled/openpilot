#!/usr/bin/env python3
"""
Plant-fit-focused HTML report.

Renders, for each dongle in plant_fit.json:

  - Per-speed-anchor fitted (G, bias, Td, R², LOO RMSE, identity RMSE,
    improvement vs identity, deployable flag).
  - The deployable correction curve as a plot.
  - A "what does this car need" verdict.

And a fleet-level summary:
  - How many (dongle, speed_bin) cells are deployable.
  - Cross-car agreement on G and bias at each speed anchor.
  - Final recommendation: deploy per-dongle, deploy fleet-wide, or do not
    deploy.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPEED_ANCHORS_KPH = (20, 40, 60, 80, 100, 120, 140)


def _img(fig, max_width=900) -> str:
  buf = io.BytesIO()
  fig.savefig(buf, format="webp", bbox_inches="tight", dpi=120)
  plt.close(fig)
  b64 = base64.b64encode(buf.getvalue()).decode()
  return f"<img src='data:image/webp;base64,{b64}' style='width:100%; max-width:{max_width}px;'>"


def _gain_table(dongle: str, bins: dict) -> str:
  rows = []
  rows.append("<tr><th>v (km/h)</th><th>N samples</th><th>routes</th>"
              "<th>identity RMSE</th><th>chosen</th>"
              "<th>G</th><th>bias (rad/m)</th><th>Td (ms)</th>"
              "<th>in-sample R²</th><th>LOO RMSE</th>"
              "<th>improvement</th><th>deployable</th></tr>")
  for v in SPEED_ANCHORS_KPH:
    b = bins.get(f"{v}kph")
    if b is None:
      rows.append(f"<tr><td>{v}</td><td colspan=11>(missing)</td></tr>")
      continue
    chosen = b.get("chosen_model", "none")
    if chosen == "arx":
      G = b.get("arx_K"); bias = 0.0
      Td = b.get("arx_Td_s"); r2 = b.get("arx_in_sample_r2")
      loo = b.get("arx_loo_rmse")
    elif chosen == "static":
      G = b.get("static_G"); bias = b.get("static_bias")
      Td = b.get("static_Td_s"); r2 = b.get("static_in_sample_r2")
      loo = b.get("static_loo_rmse")
    else:
      G = None; bias = None; Td = None; r2 = None; loo = None
    iden = b.get("identity_rmse")
    improvement = "—"
    if isinstance(loo, (int, float)) and isinstance(iden, (int, float)) and iden:
      improvement = f"{100 * (1 - loo / iden):.1f}%"
    cells = [
      str(v),
      str(b.get("n_samples", 0)),
      str(b.get("n_routes", 0)),
      f"{iden:.5f}" if isinstance(iden, (int, float)) else "—",
      chosen,
      f"{G:.3f}" if isinstance(G, (int, float)) else "—",
      f"{bias:+.5f}" if isinstance(bias, (int, float)) else "—",
      f"{(Td or 0) * 1000:.0f}" if Td is not None else "—",
      f"{r2:.2f}" if isinstance(r2, (int, float)) else "—",
      f"{loo:.5f}" if isinstance(loo, (int, float)) else "—",
      improvement,
      ("<b style='color:#2c2'>YES</b>" if b.get("deployable") else
       f"<span style='color:#a33' title='{escape(b.get('reject_reason',''))}'>no</span>"),
    ]
    rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
  return "<table>" + "".join(rows) + "</table>"


def _correction_curve(dongle: str, bins: dict) -> str:
  """Plot (G, bias) per speed anchor for this car. Deployable anchors only."""
  vs = []; Gs = []; bs = []; iden = []
  for v in SPEED_ANCHORS_KPH:
    b = bins.get(f"{v}kph", {})
    if not b.get("deployable"):
      continue
    chosen = b.get("chosen_model")
    if chosen == "arx":
      G = b.get("arx_K"); bias = 0.0
    else:
      G = b.get("static_G"); bias = b.get("static_bias")
    if G is None or bias is None:
      continue
    vs.append(v); Gs.append(G); bs.append(bias)
    iden.append(b.get("identity_rmse") or 0.0)
  if not vs:
    return f"<p><i>No deployable speed anchors for {escape(dongle)}.</i></p>"
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
  ax1.plot(vs, Gs, "o-")
  ax1.axhline(1.0, color="gray", ls="--", alpha=0.5, label="identity (G=1)")
  ax1.set_xlabel("speed (km/h)"); ax1.set_ylabel("fitted plant gain G")
  ax1.set_ylim(0.5, 1.5); ax1.grid(True, alpha=0.3); ax1.legend()
  ax1.set_title(f"{dongle} — gain")
  ax2.plot(vs, bs, "s-", color="#cc4")
  ax2.axhline(0.0, color="gray", ls="--", alpha=0.5)
  ax2.set_xlabel("speed (km/h)"); ax2.set_ylabel("fitted plant bias (rad/m)")
  ax2.grid(True, alpha=0.3)
  ax2.set_title(f"{dongle} — bias")
  return _img(fig)


def _verdict(dongle: str, bins: dict) -> str:
  total = len(SPEED_ANCHORS_KPH)
  dep = sum(1 for v in SPEED_ANCHORS_KPH if bins.get(f"{v}kph", {}).get("deployable"))
  dep_hwy = sum(1 for v in SPEED_ANCHORS_KPH
                if v >= 80 and bins.get(f"{v}kph", {}).get("deployable"))
  total_hwy = sum(1 for v in SPEED_ANCHORS_KPH if v >= 80)
  if dep_hwy == 0:
    verdict = (f"<b>NOT DEPLOYABLE for highway.</b> 0 of {total_hwy} highway "
               "speed bins met the deployment criterion. "
               "Most likely reason: insufficient data at highway speeds for this car.")
  elif dep_hwy == total_hwy:
    verdict = (f"<b style='color:#2c2'>DEPLOYABLE for highway.</b> "
               f"{dep_hwy} of {total_hwy} highway bins fitted and LOO-validated. "
               "Use apply.py:EpsCorrection.from_json(dongle=<this>).")
  else:
    verdict = (f"<b>Partially deployable.</b> {dep_hwy} of {total_hwy} highway bins "
               "passed. Below those speeds the correction passes through unchanged.")
  return f"<div class='verdict'>{verdict}<br>"\
         f"Total: {dep}/{total} bins deployable across all speeds.</div>"


def _fleet_summary(data: dict) -> str:
  """Cross-car agreement on G and bias at each speed anchor."""
  rows = []
  rows.append("<tr><th>v (km/h)</th><th>cars with deployable fit</th>"
              "<th>median G</th><th>spread (max-min)</th>"
              "<th>median bias</th><th>spread bias</th>"
              "<th>fleet-wide candidate?</th></tr>")
  for v in SPEED_ANCHORS_KPH:
    Gs = []; biases = []
    for dongle, bins in data.items():
      b = bins.get(f"{v}kph", {})
      if not b.get("deployable"):
        continue
      chosen = b.get("chosen_model")
      if chosen == "arx":
        G = b.get("arx_K"); bias = 0.0
      else:
        G = b.get("static_G"); bias = b.get("static_bias")
      if G is None or bias is None:
        continue
      Gs.append(G); biases.append(bias)
    if not Gs:
      rows.append(f"<tr><td>{v}</td><td>0</td><td colspan=5>—</td></tr>")
      continue
    g_arr = np.array(Gs); b_arr = np.array(biases)
    fleet_ok = (len(Gs) >= 3) and (g_arr.max() - g_arr.min() <= 0.10) \
               and (b_arr.max() - b_arr.min() <= 1e-4)
    rows.append(
      f"<tr><td>{v}</td><td>{len(Gs)}</td>"
      f"<td>{np.median(g_arr):.3f}</td><td>{g_arr.max()-g_arr.min():.3f}</td>"
      f"<td>{np.median(b_arr):+.5f}</td><td>{b_arr.max()-b_arr.min():.5f}</td>"
      f"<td>{'YES' if fleet_ok else 'no'}</td></tr>"
    )
  return "<table>" + "".join(rows) + "</table>"


def render(plant_fit_path: str, out_path: str) -> None:
  with open(plant_fit_path) as f:
    data = json.load(f)
  cars = list(data.keys())
  n_cars = len(cars)

  parts = []
  parts.append("""<!doctype html>
<html><head><meta charset='utf-8'><title>EPS Plant Fit — ID4_MK1</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto;
         padding: 0 1em; color: #222; }
  h1 { border-bottom: 2px solid #555; }
  h2 { margin-top: 2em; border-bottom: 1px solid #aaa; }
  h3 { background: #f5f5f5; padding: 0.3em 0.6em; }
  table { border-collapse: collapse; margin: 1em 0; }
  td, th { padding: 4px 10px; border: 1px solid #ccc; font-size: 0.9em; text-align: right; }
  th { background: #f3f3f3; text-align: center; }
  .verdict { background: #fff7d6; padding: 0.6em 1em; border-left: 4px solid #cc9900;
             margin: 1em 0; }
  .warn { background: #ffe4e0; padding: 0.6em 1em; border-left: 4px solid #c33;
          margin: 1em 0; }
  .meta { color: #555; font-size: 0.9em; }
</style></head><body>
""")
  parts.append("<h1>EPS Plant Fit — VOLKSWAGEN_ID4_MK1</h1>")
  parts.append(f"<div class='meta'>cars in fit: {n_cars}, "
               f"speed anchors: {', '.join(map(str, SPEED_ANCHORS_KPH))} km/h, "
               f"per-bin LOO-CV across routes within each car.</div>")

  # Total deployable count
  n_dep = sum(1 for bins in data.values()
              for b in bins.values() if b.get("deployable"))
  n_hwy = sum(1 for bins in data.values()
              for k, b in bins.items() if int(k.replace("kph", "")) >= 80 and b.get("deployable"))
  parts.append(f"<h2>Headline</h2>"
               f"<div class='verdict'><b>{n_dep}</b> total deployable (car, speed_bin) cells. "
               f"<b>{n_hwy}</b> highway-speed (≥80 km/h) cells deployable. "
               f"<br>Cross-car distribution shown below.</div>")

  parts.append("<h2>Fleet-level cross-car summary</h2>")
  parts.append(_fleet_summary(data))

  parts.append("<h2>Per-car details</h2>")
  # Sort by total deployable cells, then dongle id
  car_order = sorted(cars, key=lambda d: (
    -sum(1 for b in data[d].values() if b.get("deployable")), d))
  for dongle in car_order:
    bins = data[dongle]
    n_dep_car = sum(1 for b in bins.values() if b.get("deployable"))
    total_segs = sum(b.get("n_routes", 0) for b in bins.values())
    parts.append(f"<h3>{escape(dongle)} — {n_dep_car}/{len(SPEED_ANCHORS_KPH)} deployable</h3>")
    parts.append(_verdict(dongle, bins))
    parts.append(_correction_curve(dongle, bins))
    parts.append(_gain_table(dongle, bins))

  parts.append("</body></html>")
  with open(out_path, "w") as f:
    f.write("".join(parts))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--plant-fit", default="plant_fit.json")
  p.add_argument("--out", default="eps_report.html")
  args = p.parse_args()
  render(args.plant_fit, args.out)
  print(f"[eps_report] wrote {args.out}", flush=True)


if __name__ == "__main__":
  sys.exit(main())
