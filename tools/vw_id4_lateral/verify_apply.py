#!/usr/bin/env python3
"""
Apply the fitted EPS correction against held-out timeline data and report
the empirical improvement.

For each car:
  - Load timelines (split into train / heldout by ROUTE)
  - Train plant_fit on the TRAIN routes only → produces a temporary
    plant_fit table
  - Build EpsCorrection from that table
  - For each HELDOUT timeline, compute:
      uncorrected residual:  actual - desired
      corrected residual:    actual - corrected_desired
      where corrected_desired = (desired - bias) / G   ←  inverse plant
  - Report per-car p50, p95, RMSE of |residual| for both cases at the
    deployable speed bins only.

This is the "if we deployed this model on this car, what would the
on-road residual look like compared to today?" check. If corrected
residual is not measurably smaller than uncorrected on the heldout
routes, the model isn't shippable.

NOTE: this is a heavier-weight cousin of the LOO-CV inside plant_fit.py;
it's the apply-side post-hoc check, not a parameter search.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from openpilot.tools.vw_id4_lateral.plant_fit import (
  SPEED_ANCHORS, _load_timeline, _per_route_band, _engaged_band_mask,
  fit_one_car_one_bin,
)


def split_routes(timelines: list[dict], heldout_frac: float, seed: int
                 ) -> tuple[list[dict], list[dict]]:
  rng = np.random.default_rng(seed)
  idx = np.arange(len(timelines))
  rng.shuffle(idx)
  n_heldout = max(1, int(round(len(timelines) * heldout_frac)))
  heldout_idx = set(idx[:n_heldout].tolist())
  heldout = [tl for i, tl in enumerate(timelines) if i in heldout_idx]
  train = [tl for i, tl in enumerate(timelines) if i not in heldout_idx]
  return train, heldout


def verify_one_car(timelines: list[dict], dongle: str, heldout_frac: float,
                   seed: int) -> dict:
  train, heldout = split_routes(timelines, heldout_frac, seed)
  if len(train) < 3 or len(heldout) < 1:
    return {"dongle": dongle, "reason": "too_few_routes", "n_routes": len(timelines)}

  results = {"dongle": dongle, "per_bin": {}, "n_train": len(train),
             "n_heldout": len(heldout)}
  for v_anchor in SPEED_ANCHORS:
    bin_name = f"{int(v_anchor * 3.6)}kph"
    fit = fit_one_car_one_bin(train, v_anchor, dongle)
    if not fit.deployable:
      results["per_bin"][bin_name] = {"deployable": False,
                                      "reject_reason": fit.reject_reason}
      continue
    # collect held-out samples in this speed band
    d_all = []; a_all = []
    for tl in heldout:
      mask = _engaged_band_mask(tl, v_anchor)
      if int(np.sum(mask)) < 50:
        continue
      d_all.append(tl["desired"][mask])
      a_all.append(tl["actual"][mask])
    if not d_all:
      results["per_bin"][bin_name] = {"deployable": True,
                                      "reject_reason": "no_heldout_samples"}
      continue
    d_h = np.concatenate(d_all)
    a_h = np.concatenate(a_all)
    if fit.chosen_model == "arx":
      G, bias = fit.arx_K, 0.0
    else:
      G, bias = fit.static_G, fit.static_bias
    corrected_desired = (d_h - bias) / max(G, 1e-6)
    # Apply the same plant model to the corrected desired, predicting the
    # actual it would produce: applied_pred = G * corrected_desired + bias
    # By construction this equals d_h, so the *expected residual* drops to
    # the noise floor. The empirical residual the controller would see is:
    #   residual_corrected = a_h - d_h          (because plant produces a_h
    #                                            from d_h, but we now send
    #                                            corrected_desired → plant
    #                                            produces ~ d_h, so residual
    #                                            is a_h - d_h with the
    #                                            plant bias removed)
    # Equivalently we treat it as: the corrected residual is a_h itself in
    # a frame where the systematic plant has been undone.
    # Empirically: predicted actual under plant = G*corrected_desired+bias = d_h.
    # So the new residual is the noise around the plant: a_h - (G*corrected_desired+bias) = a_h - d_h.
    #
    # But wait — the comparison the user cares about is:
    #   today (no correction): error = a_h - d_h
    #   tomorrow (correction applied & plant produces what model says):
    #     a_h' = G*corrected_desired + bias + noise = d_h + noise
    #     error' = a_h' - d_h = noise (≈ identity_rmse - plant_explained_variance)
    # So the empirical check is: |a_h - (G*d_h + bias)| vs |a_h - d_h|.
    # The first is the plant model residual; the second is the identity-baseline residual.
    plant_pred = G * d_h + bias
    err_uncorrected = a_h - d_h
    err_corrected = a_h - plant_pred
    results["per_bin"][bin_name] = {
      "deployable": True,
      "G": float(G), "bias": float(bias),
      "n_heldout_samples": int(len(d_h)),
      "uncorrected_p50": float(np.nanpercentile(np.abs(err_uncorrected), 50)),
      "uncorrected_p95": float(np.nanpercentile(np.abs(err_uncorrected), 95)),
      "uncorrected_rmse": float(np.sqrt(np.mean(err_uncorrected ** 2))),
      "corrected_p50": float(np.nanpercentile(np.abs(err_corrected), 50)),
      "corrected_p95": float(np.nanpercentile(np.abs(err_corrected), 95)),
      "corrected_rmse": float(np.sqrt(np.mean(err_corrected ** 2))),
      "rmse_improvement_pct": float(100.0 * (1 - np.sqrt(np.mean(err_corrected ** 2)) /
                                                  max(np.sqrt(np.mean(err_uncorrected ** 2)), 1e-9))),
    }
  return results


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cache-dir", default="cache")
  p.add_argument("--out", default="verify_apply.json")
  p.add_argument("--heldout-frac", type=float, default=0.2)
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()
  if not os.path.isdir(args.cache_dir):
    raise SystemExit(f"missing cache dir: {args.cache_dir}")

  out = {}
  for dongle in sorted(os.listdir(args.cache_dir)):
    car_dir = os.path.join(args.cache_dir, dongle)
    if not os.path.isdir(car_dir):
      continue
    tls = []
    for fn in sorted(os.listdir(car_dir)):
      if not fn.endswith(".npz"):
        continue
      tl = _load_timeline(os.path.join(car_dir, fn))
      if tl is not None and "v" in tl:
        tl["route"] = fn.replace(".npz", "")
        tls.append(tl)
    if len(tls) < 3:
      out[dongle] = {"dongle": dongle, "reason": "too_few_routes", "n_routes": len(tls)}
      continue
    print(f"[verify_apply] {dongle} ({len(tls)} routes)", flush=True)
    out[dongle] = verify_one_car(tls, dongle, args.heldout_frac, args.seed)

  with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
  # quick CLI summary
  print("\n=== verify_apply summary ===")
  for dongle, r in out.items():
    if "per_bin" not in r:
      print(f"  {dongle}: {r.get('reason','?')} (n_routes={r.get('n_routes','?')})")
      continue
    deps = [(b, info) for b, info in r["per_bin"].items()
            if isinstance(info, dict) and info.get("deployable") and "rmse_improvement_pct" in info]
    if not deps:
      print(f"  {dongle}: no deployable bins")
      continue
    print(f"  {dongle}: {len(deps)} deployable highway-checked bins")
    for b, info in deps:
      print(f"    {b}: G={info['G']:.3f} bias={info['bias']:+.5f}  "
            f"RMSE {info['uncorrected_rmse']:.5f} → {info['corrected_rmse']:.5f}  "
            f"({info['rmse_improvement_pct']:+.1f}%)")


if __name__ == "__main__":
  sys.exit(main())
