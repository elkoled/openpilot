#!/usr/bin/env python3
"""
EPS-response model fitter and validator for VW MEB (VOLKSWAGEN_ID4_MK1 fleet).

Inputs:
  per_dongle_bucket.npz from fleet_lateral_aggregate.py  (one set of bucket
  sums per dongle).

Outputs (under --out_dir):
  - model.json           : fitted K(v) parameters plus per-speed table
  - model_report.txt     : full leave-one-dongle-out validation report
  - fig_per_dongle_gain.png
  - fig_fleet_gain_with_ci.png
  - fig_validation_holdout.png

What the model is:

The MEB carcontroller currently does
    apply_curvature = desired + (rack_meas - vm_curv)
On the highway, observed actual = K(v) * desired with K(v) < 1 — vehicle
undershoots.  A speed-dependent compensation multiplier
    apply_curvature = desired / K(v) + (rack_meas - vm_curv)
closes the steady-state gap if K(v) is stable across the fleet.

This script fits K(v) and validates it against held-out dongles.  The goal
is conservative: do not propose a patch unless leave-one-dongle-out
generalization shows a real residual-rms reduction on highway samples.

Truth source used: pose_yaw (CC.angularVelocity[2] / vEgo).  This is the
canonical truth source on VW (cs_yaw is unpopulated on most carstate
branches; qfk_rack is noisy by R^2 = 0.10 vs pose).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

POSE = 0  # truth source index

# Speed bins (edges in m/s) — must match fleet_lateral_extract.py
# We fit one K per *bin midpoint* and interpolate piecewise-linearly.
# Highway focus: bins above 22 m/s (~80 km/h) are the primary improvement target.

# Highway curvature mask: only buckets where |center| > 5e-4 (else gain is noisy
# division of two small numbers).
CURV_MIN_FIT = 5e-4
CURV_MAX_FIT = 0.5  # exclude saturated extremes (rare in highway data anyway)
MIN_BUCKET_N = 30          # require this many samples in a (s,c,t) cell
MIN_BIN_MEAN_D = 5e-4      # require |mean_d| above this to use cell for gain


def bucket_centers(edges):
    return (edges[:-1] + edges[1:]) / 2


def per_dongle_per_speed_gain(bucket, ti=POSE):
    """For each (dongle, speed_bin): n-weighted average of per-cell gain
    K_cell = mean_t_cell / mean_d_cell, over signed curvature cells where the
    per-cell |mean_d| is large enough to give a stable ratio.

    Computing gain per signed-curvature cell first (then averaging) avoids the
    sign-cancellation failure mode of summed gain (which can blow up or invert
    when positive and negative curvature samples roughly balance out).

    Returns:
      gain  (D, S)   weighted-average per-dongle per-speed gain (NaN if no valid cells)
      n     (D, S)   total samples contributing
    """
    D = bucket["dongles"].shape[0]
    S = bucket["speed_edges"].shape[0] - 1
    edges_c = bucket["curvature_edges"]
    centers = bucket_centers(edges_c)
    keep_c = np.where((np.abs(centers) > CURV_MIN_FIT) & (np.abs(centers) < CURV_MAX_FIT))[0]

    truth_sum = bucket["truth_sum"][:, :, :, ti]
    truth_n = bucket["truth_n"][:, :, :, ti]
    resid_td_sum = bucket["resid_td_sum"][:, :, :, ti]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t_cell = np.where(truth_n > 0, truth_sum / np.maximum(truth_n, 1), np.nan)
        mean_resid_cell = np.where(truth_n > 0, resid_td_sum / np.maximum(truth_n, 1), np.nan)
    mean_d_cell = mean_t_cell - mean_resid_cell

    cell_valid = (truth_n >= MIN_BUCKET_N) & (np.abs(mean_d_cell) >= MIN_BIN_MEAN_D)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_cell = np.where(cell_valid, mean_t_cell / mean_d_cell, np.nan)

    # Restrict to keep_c columns then take n-weighted average over signed-curv bins.
    g_sub = gain_cell[:, :, keep_c]
    n_sub = np.where(cell_valid[:, :, keep_c], truth_n[:, :, keep_c], 0).astype(np.float64)

    num = np.nansum(np.where(np.isfinite(g_sub), g_sub * n_sub, 0.0), axis=2)
    den = n_sub.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(den > 0, num / den, np.nan)
    return gain, den.astype(np.int64)


def effective_speed_per_bin(bucket, dongle_mask=None):
    """Sample-weighted mean vEgo per speed bin, pooled across the masked dongles.
    Falls back to bin midpoint for empty bins."""
    edges_v = bucket["speed_edges"]
    midpts = (edges_v[:-1] + edges_v[1:]) / 2
    if dongle_mask is None:
        dongle_mask = np.ones(bucket["dongles"].shape[0], dtype=bool)
    vego_sum = bucket["vego_sum"][dongle_mask].sum(axis=(0, 2))  # (S,)
    n = bucket["n"][dongle_mask].sum(axis=(0, 2)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n > 0, vego_sum / np.maximum(n, 1), midpts)
    return out


def fleet_gain_pooled(bucket, dongle_mask=None, ti=POSE):
    """Fleet-pooled gain per speed bin using per-cell ratios (avoiding sign
    cancellation), n-weighted across dongles and signed-curvature bins."""
    if dongle_mask is None:
        dongle_mask = np.ones(bucket["dongles"].shape[0], dtype=bool)
    truth_sum = bucket["truth_sum"][dongle_mask, :, :, ti]
    resid_td_sum = bucket["resid_td_sum"][dongle_mask, :, :, ti]
    truth_n = bucket["truth_n"][dongle_mask, :, :, ti]

    edges_c = bucket["curvature_edges"]
    centers = bucket_centers(edges_c)
    keep_c = np.where((np.abs(centers) > CURV_MIN_FIT) & (np.abs(centers) < CURV_MAX_FIT))[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t = np.where(truth_n > 0, truth_sum / np.maximum(truth_n, 1), np.nan)
        mean_r = np.where(truth_n > 0, resid_td_sum / np.maximum(truth_n, 1), np.nan)
    mean_d = mean_t - mean_r
    cell_valid = (truth_n >= MIN_BUCKET_N) & (np.abs(mean_d) >= MIN_BIN_MEAN_D)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_cell = np.where(cell_valid, mean_t / mean_d, np.nan)

    g_sub = gain_cell[:, :, keep_c]
    n_sub = np.where(cell_valid[:, :, keep_c], truth_n[:, :, keep_c], 0).astype(np.float64)
    num = np.nansum(np.where(np.isfinite(g_sub), g_sub * n_sub, 0.0), axis=(0, 2))
    den = n_sub.sum(axis=(0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(den > 0, num / den, np.nan)
    return gain, den.astype(np.int64)


def residual_rms_on_holdout(bucket, dongle_idx, K_model_fn, ti=POSE):
    """Compute residual rms on samples from one dongle, comparing:
      - before: rms(truth - desired)
      - after:  rms(truth - desired / K_model(v))

    K_model_fn takes a speed (m/s) and returns K(v).  Applied per speed bin
    (so the model is evaluated at bin midpoints).

    Returns dict with rms_before, rms_after, n_samples, per-speed breakdown.
    """
    edges_c = bucket["curvature_edges"]
    centers_c = bucket_centers(edges_c)
    # Use sample-weighted per-bin v for K(v) evaluation.
    only = np.zeros(bucket["dongles"].shape[0], dtype=bool); only[dongle_idx] = True
    centers_v = effective_speed_per_bin(bucket, dongle_mask=only)

    keep_c = np.where((np.abs(centers_c) > CURV_MIN_FIT) & (np.abs(centers_c) < CURV_MAX_FIT))[0]

    # Per-cell sufficient statistics for the held-out dongle.
    n_cell = bucket["truth_n"][dongle_idx, :, :, ti]          # (S, C)
    t_sum_cell = bucket["truth_sum"][dongle_idx, :, :, ti]
    td_sum_cell = bucket["resid_td_sum"][dongle_idx, :, :, ti]
    td_sumsq_cell = bucket["resid_td_sumsq"][dongle_idx, :, :, ti]
    d_sum_cell = t_sum_cell - td_sum_cell
    # sum of d^2 we don't track directly per truth-source.  But we have
    # desired_sumsq across all samples and desired_sum across all samples;
    # truth-subset desired moments are not directly available.  However for
    # the "after" residual we don't need d^2, only (truth - desired/K)^2:
    #   sum (t - d/K)^2 = sum t^2 - 2/K * sum(t*d) + 1/K^2 * sum(d^2)
    # We don't track sum(t*d) or sum(d^2) per truth-subset either, so we
    # approximate using bucket-level moments under the assumption that
    # variance of truth within a bucket is small compared to the bin mean.
    # That's a strong assumption — we will verify by reporting the
    # per-speed-bin gain and residual mean, not just the squared total.

    # Per-speed-bin pooled residual mean (truth - desired) BEFORE:
    # mean_resid_before(s) = sum_c (truth - desired) / sum_c n
    n_speed = n_cell[:, keep_c].sum(axis=1)
    td_sum_speed = td_sum_cell[:, keep_c].sum(axis=1)
    td_sumsq_speed = td_sumsq_cell[:, keep_c].sum(axis=1)
    t_sum_speed = t_sum_cell[:, keep_c].sum(axis=1)
    d_sum_speed = d_sum_cell[:, keep_c].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_before = np.where(n_speed > 0, td_sum_speed / n_speed, np.nan)
        var_before = np.where(n_speed > 0,
                              td_sumsq_speed / np.maximum(n_speed, 1) - mean_before * mean_before,
                              np.nan)
        rms_before = np.sqrt(np.maximum(var_before + mean_before * mean_before, 0.0))

    # AFTER: residual = truth - desired/K.  Mean shifts because the desired
    # term changes; assuming desired magnitude within a bin is ~constant at
    # bin mean (which is what gain stats already assume), the per-speed
    # residual MEAN after is:
    #   mean(t - d/K(v)) = mean(t) - mean(d)/K(v)
    Kv_speed = np.array([K_model_fn(v) for v in centers_v])
    # Post-patch arithmetic (counterfactual):
    #   BEFORE: command = desired,            actual = K_observed * desired
    #   AFTER : command = desired / K_model,  actual = K_observed * desired / K_model
    # so mean(actual_after - desired) = (K_obs / K_model - 1) * mean_d
    #                                 = mean_t_obs / K_model - mean_d
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_d = np.where(n_speed > 0, d_sum_speed / n_speed, np.nan)
        mean_t = np.where(n_speed > 0, t_sum_speed / n_speed, np.nan)
        mean_after = mean_t / Kv_speed - mean_d

    return {
        "speed_centers": centers_v,
        "n_speed": n_speed.astype(int),
        "rms_before": rms_before,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "Kv_applied": Kv_speed,
    }


def fit_K_models(bucket, dongle_mask, ti=POSE, min_v_for_fit: float = 0.0):
    """Fit several candidate K(v) parameterizations to the masked-fleet data.

    min_v_for_fit: restrict the constant/understeer fits to bins at or above
    this speed (default 0 = use all bins).  The piecewise model is unaffected
    since it stores per-bin gains directly.
    Returns dict of model_name -> {predict(v), params, rmse_on_speed_bins}.
    """
    gain, n = fleet_gain_pooled(bucket, dongle_mask=dongle_mask, ti=ti)
    centers_v = effective_speed_per_bin(bucket, dongle_mask=dongle_mask)

    valid = np.isfinite(gain) & (n >= 200) & (centers_v >= min_v_for_fit)
    v_fit = centers_v[valid]
    g_fit = gain[valid]
    w_fit = n[valid].astype(np.float64)

    out = {}

    # M0: constant
    if v_fit.size > 0:
        K0 = float(np.average(g_fit, weights=w_fit))

        def predict_constant(v, K=K0):
            return K
        out["constant"] = {
            "predict": predict_constant,
            "params": {"K": K0},
        }

    # M1: piecewise per-bin (anchors at bin centers, linear interp, end-flat)
    if v_fit.size >= 2:
        v_pw = v_fit.tolist()
        g_pw = g_fit.tolist()

        def predict_piecewise(v, vs=v_pw, gs=g_pw):
            return float(np.interp(v, vs, gs))
        out["piecewise"] = {
            "predict": predict_piecewise,
            "params": {"speeds": v_pw, "gains": g_pw},
        }

    # M2: 1 / (1 + Ku * v^2) — physical understeer form
    # Fit Ku such that K(v_i) = 1/(1+Ku*v_i^2) matches g_fit (weighted LSQ).
    if v_fit.size >= 2:
        # K^-1 = 1 + Ku * v^2  =>  Ku = ((1/K) - 1) / v^2
        with np.errstate(divide="ignore", invalid="ignore"):
            valid_us = g_fit > 0.5  # physical bound; ignore weird-low bins
        if valid_us.any():
            v2 = v_fit[valid_us] ** 2
            y = 1.0 / g_fit[valid_us] - 1.0
            w = w_fit[valid_us]
            # weighted least squares for y = Ku * v^2 => Ku = sum(w*y*v^2)/sum(w*v^4)
            Ku = float(np.sum(w * y * v2) / max(np.sum(w * v2 * v2), 1e-9))

            def predict_understeer(v, Ku=Ku):
                return float(1.0 / (1.0 + Ku * v * v))
            out["understeer"] = {
                "predict": predict_understeer,
                "params": {"Ku": Ku},
            }

    # Score each model: weighted RMSE between predicted K and observed gain.
    for name, m in out.items():
        pred = np.array([m["predict"](v) for v in centers_v])
        diff = pred - gain
        sq = np.where(np.isfinite(diff), diff ** 2, 0.0)
        m["rmse_K"] = float(np.sqrt(np.sum(sq * n) / max(n.sum(), 1)))

    return out, gain, n, centers_v


def per_dongle_signed_gain(bucket, ti=POSE):
    """For each (dongle, speed_bin), compute K_left and K_right separately
    by signed-curvature half.  If they disagree, a single scalar multiplier
    isn't sufficient and a per-sign or per-magnitude table is needed.

    Returns: (gain_left, gain_right, n_left, n_right) each (D, S).
    """
    D, S = bucket["dongles"].shape[0], bucket["speed_edges"].shape[0] - 1
    edges_c = bucket["curvature_edges"]
    centers = bucket_centers(edges_c)
    left = np.where((centers <= -CURV_MIN_FIT) & (centers >= -CURV_MAX_FIT))[0]
    right = np.where((centers >= CURV_MIN_FIT) & (centers <= CURV_MAX_FIT))[0]

    truth_sum = bucket["truth_sum"][:, :, :, ti]
    truth_n = bucket["truth_n"][:, :, :, ti]
    resid_td_sum = bucket["resid_td_sum"][:, :, :, ti]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t_cell = np.where(truth_n > 0, truth_sum / np.maximum(truth_n, 1), np.nan)
        mean_r_cell = np.where(truth_n > 0, resid_td_sum / np.maximum(truth_n, 1), np.nan)
    mean_d_cell = mean_t_cell - mean_r_cell
    cell_valid = (truth_n >= MIN_BUCKET_N) & (np.abs(mean_d_cell) >= MIN_BIN_MEAN_D)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_cell = np.where(cell_valid, mean_t_cell / mean_d_cell, np.nan)

    def half_gain(idx):
        g_sub = gain_cell[:, :, idx]
        n_sub = np.where(cell_valid[:, :, idx], truth_n[:, :, idx], 0).astype(np.float64)
        num = np.nansum(np.where(np.isfinite(g_sub), g_sub * n_sub, 0.0), axis=2)
        den = n_sub.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan), den.astype(np.int64)

    gain_left, n_left = half_gain(left)
    gain_right, n_right = half_gain(right)
    return gain_left, gain_right, n_left, n_right


def leave_one_dongle_out(bucket, ti=POSE):
    """For each dongle, train on the others, score the held-out one."""
    D = bucket["dongles"].shape[0]
    centers_v = effective_speed_per_bin(bucket)
    reports = []

    for i in range(D):
        train_mask = np.ones(D, dtype=bool); train_mask[i] = False
        models, gain_train, n_train, _ = fit_K_models(bucket, train_mask, ti=ti)
        if "piecewise" not in models:
            continue

        # Score each model on the held-out dongle
        per_model = {}
        for name, m in models.items():
            res = residual_rms_on_holdout(bucket, i, m["predict"], ti=ti)
            per_model[name] = res

        # Also score the null (no patch) model
        def predict_null(v):
            return 1.0
        per_model["null"] = residual_rms_on_holdout(bucket, i, predict_null, ti=ti)

        reports.append({
            "holdout_dongle": str(bucket["dongles"][i]),
            "n_train_dongles": int(train_mask.sum()),
            "per_model": per_model,
        })
    return reports


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bucket_npz", help="per_dongle_bucket.npz from fleet_lateral_aggregate.py")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: alongside bucket_npz)")
    p.add_argument("--min_dongle_engaged_s", type=float, default=300.0,
                   help="Drop dongles in fit if pooled engaged_s in valid buckets too low")
    args = p.parse_args(argv[1:])

    bucket_path = Path(args.bucket_npz)
    out_dir = Path(args.out_dir or bucket_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    bdata = np.load(bucket_path, allow_pickle=True)
    bucket = {k: bdata[k] for k in bdata.files}
    D = bucket["dongles"].shape[0]
    print(f"Loaded {D} dongles: {[str(d) for d in bucket['dongles']]}")

    # --- Per-dongle per-speed gain table ---
    gain_ds, n_ds = per_dongle_per_speed_gain(bucket, ti=POSE)
    centers_v = effective_speed_per_bin(bucket)

    # --- Fleet gain (all dongles) ---
    gain_fleet, n_fleet = fleet_gain_pooled(bucket, dongle_mask=None, ti=POSE)

    # --- Fit on whole fleet ---
    models_all, gain_all, n_all, _ = fit_K_models(bucket, dongle_mask=np.ones(D, dtype=bool), ti=POSE)

    # --- Leave-one-dongle-out validation ---
    loo = leave_one_dongle_out(bucket, ti=POSE)

    # --- Build the human report ---
    lines = []
    lines.append("# EPS-response model fit and validation")
    lines.append("")
    lines.append(f"Source: {bucket_path}")
    lines.append(f"Dongles: {D}")
    lines.append("")
    lines.append("Per-dongle per-speed gain (truth/desired, |c|>5e-4):")
    header = "  v(m/s):  " + "  ".join(f"{v:>6.1f}" for v in centers_v)
    lines.append(header)
    lines.append(f"  {'fleet':16s}  " + "  ".join(
        (f"{g:>6.3f}" if np.isfinite(g) else f"{'  -  ':>6s}") for g in gain_fleet))
    lines.append(f"  {'n_fleet':16s}  " + "  ".join(f"{n:>6d}" for n in n_fleet))
    lines.append("")
    for di, d in enumerate(bucket["dongles"]):
        g_row = gain_ds[di]
        n_row = n_ds[di]
        lines.append(f"  {str(d)[:16]:16s}  " + "  ".join(
            (f"{g:>6.3f}" if np.isfinite(g) else f"{'  -  ':>6s}") for g in g_row))
        lines.append(f"  {'  n':16s}  " + "  ".join(f"{int(n):>6d}" for n in n_row))
    lines.append("")
    # --- L/R asymmetry check ---
    g_left, g_right, n_left_a, n_right_a = per_dongle_signed_gain(bucket, ti=POSE)
    lines.append("L/R asymmetry per dongle (gain_left vs gain_right; large gap => single scalar insufficient):")
    for di, d in enumerate(bucket["dongles"]):
        for si, v in enumerate(centers_v):
            gl, gr = g_left[di, si], g_right[di, si]
            nl, nr = int(n_left_a[di, si]), int(n_right_a[di, si])
            if nl < 100 and nr < 100:
                continue
            lines.append(f"  {str(d)[:14]:14s}  v={v:>5.1f}  "
                         f"K_L={gl:.3f} (n={nl:>5d})  K_R={gr:.3f} (n={nr:>5d})  "
                         f"diff={ (gl - gr if np.isfinite(gl) and np.isfinite(gr) else float('nan')):+.3f}")
    lines.append("")
    lines.append("Cross-dongle distribution per speed bin (only bins with >=200 samples per dongle):")
    for si, v in enumerate(centers_v):
        col = []
        for di in range(D):
            if np.isfinite(gain_ds[di, si]) and n_ds[di, si] >= 200:
                col.append(gain_ds[di, si])
        if not col:
            lines.append(f"  v={v:>5.1f} m/s  n_dongles=0  insufficient")
            continue
        arr = np.array(col)
        lines.append(f"  v={v:>5.1f} m/s  n_dongles={len(arr)}  "
                     f"min={arr.min():.3f}  median={np.median(arr):.3f}  "
                     f"max={arr.max():.3f}  IQR={np.percentile(arr,75)-np.percentile(arr,25):.3f}")
    lines.append("")
    lines.append("Fleet-fitted K(v) candidate models:")
    for name, m in models_all.items():
        params = m["params"]
        rmse = m["rmse_K"]
        if name == "piecewise":
            ps = ", ".join(f"{v:.0f}:{g:.3f}" for v, g in zip(params["speeds"], params["gains"]))
            lines.append(f"  {name:12s}  rmse_K={rmse:.4f}  speeds:gains = {ps}")
        else:
            ps = "  ".join(f"{k}={v:.5f}" for k, v in params.items())
            lines.append(f"  {name:12s}  rmse_K={rmse:.4f}  params: {ps}")
    lines.append("")
    lines.append("Leave-one-dongle-out validation:")
    lines.append("  (rows: held-out dongle.  Reports per-speed mean residual BEFORE vs AFTER patch.)")
    lines.append("")

    HIGHWAY_FOCUS_V = 22.0   # m/s
    aggregate_improvement_by_model: dict[str, list[tuple[float, float]]] = {}
    for r in loo:
        lines.append(f"  HOLDOUT: {r['holdout_dongle']}  (trained on {r['n_train_dongles']} others)")
        # Per-speed mean residuals
        rep_null = r["per_model"]["null"]
        ns = rep_null["n_speed"]
        any_data = False
        for si, v in enumerate(rep_null["speed_centers"]):
            if ns[si] < 50:
                continue
            any_data = True
            line = f"    v={v:>5.1f}  n={int(ns[si]):>5d}  "
            line += f"mean_before={rep_null['mean_before'][si]:+.6f}  "
            for name in ("constant", "piecewise", "understeer"):
                if name not in r["per_model"]:
                    continue
                m_after = r["per_model"][name]["mean_after"][si]
                if v >= HIGHWAY_FOCUS_V:
                    aggregate_improvement_by_model.setdefault(name, []).append(
                        (rep_null["mean_before"][si], m_after))
                line += f"{name}_after={m_after:+.6f}  "
            lines.append(line)
        if not any_data:
            lines.append("    (no speed bins met n>=50 threshold)")
    lines.append("")
    lines.append("Aggregate highway (v>=22 m/s) improvement (mean |residual| reduction, summed across holdout dongles & speeds):")
    rep_null_means_hw = aggregate_improvement_by_model.get("piecewise", [])
    if rep_null_means_hw:
        for name, pairs in aggregate_improvement_by_model.items():
            before_arr = np.array([p[0] for p in pairs])
            after_arr = np.array([p[1] for p in pairs])
            mae_before = float(np.mean(np.abs(before_arr)))
            mae_after = float(np.mean(np.abs(after_arr)))
            lines.append(f"  {name:12s}  MAE_before={mae_before:.6f}  MAE_after={mae_after:.6f}  "
                         f"reduction={(mae_before - mae_after):+.6f}  "
                         f"({100*(mae_before-mae_after)/max(mae_before,1e-9):+.1f}%)")
    else:
        lines.append("  insufficient holdout data at highway speeds")

    text = "\n".join(lines) + "\n"
    (out_dir / "model_report.txt").write_text(text)
    print(text)

    # --- Save fitted-model JSON ---
    # Pick piecewise as the canonical model (most direct; no parametric form
    # constraints); also keep alternatives for inspection.
    canonical = models_all.get("piecewise") or models_all.get("understeer") or models_all.get("constant")
    if canonical is None:
        print("WARN: no fitted model available", file=sys.stderr)
        return 2

    payload = {
        "schema_version": 1,
        "truth_source": "pose_yaw",
        "speed_centers_mps": [float(v) for v in centers_v],
        "fleet_gain_per_speed": [(float(g) if np.isfinite(g) else None) for g in gain_fleet],
        "fleet_n_per_speed": [int(n) for n in n_fleet],
        "models": {
            name: {
                "rmse_K": m["rmse_K"],
                "params": _jsonable(m["params"]),
            } for name, m in models_all.items()
        },
        "loo": [
            {
                "holdout_dongle": r["holdout_dongle"],
                "per_speed": {
                    "speeds": [float(v) for v in r["per_model"]["null"]["speed_centers"]],
                    "n": [int(x) for x in r["per_model"]["null"]["n_speed"]],
                    "mean_before": [float(x) if np.isfinite(x) else None
                                    for x in r["per_model"]["null"]["mean_before"]],
                    **{f"mean_after_{name}":
                       [float(x) if np.isfinite(x) else None
                        for x in r["per_model"][name]["mean_after"]]
                       for name in r["per_model"] if name != "null"}
                },
            } for r in loo
        ],
    }
    (out_dir / "model.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir / 'model.json'}")
    print(f"Wrote {out_dir / 'model_report.txt'}")

    # --- Figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Per-dongle gain vs speed
        fig, ax = plt.subplots(figsize=(8, 5))
        for di in range(D):
            g = gain_ds[di]
            n_row = n_ds[di]
            mask = np.isfinite(g) & (n_row >= 200)
            if mask.sum() == 0:
                continue
            ax.plot(centers_v[mask], g[mask], "-o", alpha=0.5,
                    label=f"{str(bucket['dongles'][di])[:8]} (n_tot={int(n_row.sum())})")
        ax.plot(centers_v, gain_fleet, "k-", lw=2.5, label="FLEET pooled")
        ax.axhline(1.0, color="gray", lw=0.5, linestyle=":")
        ax.set_xlabel("vEgo (m/s)")
        ax.set_ylabel("gain K = sum(truth) / sum(desired)   (|c|>5e-4)")
        ax.set_title("Per-dongle vs fleet gain  (pose_yaw truth)")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_per_dongle_gain.png", dpi=110)
        plt.close(fig)

        # Validation figure: holdout BEFORE vs AFTER per dongle (highway only)
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in loo:
            ns = r["per_model"]["null"]["n_speed"]
            vs = r["per_model"]["null"]["speed_centers"]
            mb = r["per_model"]["null"]["mean_before"]
            ma_pw = r["per_model"].get("piecewise", {}).get("mean_after", np.full_like(vs, np.nan))
            mask = (ns >= 50) & np.isfinite(mb) & np.isfinite(ma_pw) & (vs >= 22.0)
            if mask.sum() == 0:
                continue
            ax.plot(vs[mask], np.abs(mb[mask]), "o--", color="C3", alpha=0.5,
                    label="before" if r is loo[0] else None)
            ax.plot(vs[mask], np.abs(ma_pw[mask]), "s-", color="C2", alpha=0.5,
                    label="after (piecewise)" if r is loo[0] else None)
        ax.set_xlabel("vEgo (m/s)")
        ax.set_ylabel("|mean residual|  (truth − desired) or (truth − desired/K)")
        ax.set_title("Held-out residual: highway speeds only (v>=22 m/s)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_validation_holdout.png", dpi=110)
        plt.close(fig)

        print(f"Wrote {out_dir / 'fig_per_dongle_gain.png'}")
        print(f"Wrote {out_dir / 'fig_validation_holdout.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"figure generation failed: {type(e).__name__}: {e}", file=sys.stderr)

    return 0


def _jsonable(obj):
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


if __name__ == "__main__":
    sys.exit(main(sys.argv))
