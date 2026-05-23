#!/usr/bin/env python3
"""
Cross-dongle aggregator for per-route NPZ files emitted by fleet_lateral_extract.

Pools sufficient statistics by dongle, then computes the cross-dongle
*distribution* of residuals (mean / std / per-dongle bootstrap CI) — not just
the fleet mean.  The key question is: does the residual cluster by dongle?

Outputs (under --out_dir, defaults to the input dir):
  - per_dongle.tsv    : one row per dongle, with summary stats
  - per_dongle_bucket.npz : pooled sums per (dongle, speed, curvature, truth)
  - cross_dongle.tsv  : per-bucket cross-dongle mean / std / IQR
  - fleet_summary.txt : human-readable summary of the headline numbers
  - fleet_figures/    : matplotlib figures (gain-vs-speed per dongle; histogram
                         of per-dongle residual means; etc.)

Usage:
  python fleet_lateral_aggregate.py <summaries_dir> [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRUTH_NAMES = ["pose_yaw", "cs_yaw", "qfk_rack"]


def _load_route(npz_path: Path) -> dict | None:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}
        payload["header"] = json.loads(str(payload["header_json"]))
        return payload
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to load {npz_path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _bucket_residual_stats(n: np.ndarray, rsum: np.ndarray, rsumsq: np.ndarray,
                           min_n: int = 30):
    """Returns (mean, std) per cell.  Cells with fewer than min_n samples are NaN."""
    mean = np.full_like(rsum, np.nan, dtype=np.float64)
    std = np.full_like(rsum, np.nan, dtype=np.float64)
    valid = n >= min_n
    mean[valid] = rsum[valid] / n[valid]
    var = np.where(valid, rsumsq / np.maximum(n, 1) - mean * mean, 0.0)
    std[valid] = np.sqrt(np.maximum(var[valid], 0.0))
    return mean, std


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("summaries_dir", help="Directory of per-route NPZ "
                                         "(structure: <summaries_dir>/<dongle>/<route>.npz)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: alongside summaries_dir)")
    p.add_argument("--min_route_engaged_s", type=float, default=60.0,
                   help="Drop routes with engaged_s below this floor")
    p.add_argument("--min_dongle_engaged_s", type=float, default=300.0,
                   help="Drop dongles with total engaged_s below this floor")
    p.add_argument("--min_bucket_n", type=int, default=30,
                   help="Minimum samples per bucket cell to compute residual")
    args = p.parse_args(argv[1:])

    src = Path(args.summaries_dir)
    out = Path(args.out_dir or src)
    out.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src.rglob("*.npz"))
    if not npz_files:
        print(f"no NPZ files under {src}", file=sys.stderr)
        return 2

    # Pool sums by dongle.
    # We accumulate the full set of arrays from the schema.
    keys_2d = ["n", "desired_sum", "desired_sumsq",
               "vmcurv_sum", "vmcurv_sumsq",
               "hca_sent_sum", "hca_sent_sumsq", "hca_sent_n",
               "vego_sum", "roll_abs_sum"]
    keys_3d = ["truth_n", "truth_sum", "truth_sumsq",
               "resid_td_sum", "resid_td_sumsq",
               "resid_tv_sum", "resid_tv_sumsq",
               "resid_ht_sum", "resid_ht_sumsq"]

    per_dongle: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    per_dongle_meta: dict[str, dict] = defaultdict(lambda: {
        "n_routes": 0, "engaged_s": 0.0, "accepted": 0,
        "steer_ratio_sum": 0.0, "steer_ratio_n": 0,
        "stiffness_factor_sum": 0.0, "stiffness_factor_n": 0,
        "lateral_delay_sum": 0.0, "lateral_delay_n": 0,
        "car_fingerprint": "", "vins": set(), "branches": set(),
        "has_curvature_params": False, "curv_use_frac_sum": 0.0,
        "qfk_seen_routes": 0,
    })

    speed_edges = curvature_edges = truth_names = None
    n_skipped_short = n_skipped_empty = 0

    for npz_path in npz_files:
        payload = _load_route(npz_path)
        if payload is None:
            continue
        h = payload["header"]
        accepted = int(payload["n"].sum())
        if accepted == 0:
            n_skipped_empty += 1
            continue
        if h.get("engaged_s", 0.0) < args.min_route_engaged_s:
            n_skipped_short += 1
            continue

        if speed_edges is None:
            speed_edges = payload["speed_edges"].astype(np.float64)
            curvature_edges = payload["curvature_edges"].astype(np.float64)
            truth_names = list(payload["truth_names"])

        d = h["dongle_id"]
        for k in keys_2d + keys_3d:
            arr = payload[k]
            if k not in per_dongle[d]:
                per_dongle[d][k] = arr.astype(np.float64).copy() if k != "n" and k != "hca_sent_n" and k != "truth_n" else arr.astype(np.int64).copy()
            else:
                if per_dongle[d][k].dtype.kind == 'i':
                    per_dongle[d][k] += arr.astype(np.int64)
                else:
                    per_dongle[d][k] += arr.astype(np.float64)

        meta = per_dongle_meta[d]
        meta["n_routes"] += 1
        meta["engaged_s"] += float(h.get("engaged_s", 0.0))
        meta["accepted"] += accepted
        if h.get("steer_ratio_n", 0) > 0:
            meta["steer_ratio_sum"] += h["steer_ratio_mean"] * h["steer_ratio_n"]
            meta["steer_ratio_n"] += int(h["steer_ratio_n"])
        if h.get("stiffness_factor_n", 0) > 0:
            meta["stiffness_factor_sum"] += h["stiffness_factor_mean"] * h["stiffness_factor_n"]
            meta["stiffness_factor_n"] += int(h["stiffness_factor_n"])
        if h.get("lateral_delay_n", 0) > 0:
            meta["lateral_delay_sum"] += h["lateral_delay_mean"] * h["lateral_delay_n"]
            meta["lateral_delay_n"] += int(h["lateral_delay_n"])
        if h.get("car_fingerprint"):
            meta["car_fingerprint"] = h["car_fingerprint"]
        if h.get("car_vin"):
            meta["vins"].add(h["car_vin"])
        if h.get("git_branch"):
            meta["branches"].add(h["git_branch"])
        meta["has_curvature_params"] = meta["has_curvature_params"] or bool(h.get("has_curvature_params"))
        meta["curv_use_frac_sum"] += float(h.get("curvature_use_params_frac", 0.0))
        if int(h.get("qfk_bus", -1)) >= 0:
            meta["qfk_seen_routes"] += 1

    # Drop tiny dongles.
    dongles = sorted(d for d, m in per_dongle_meta.items()
                     if m["engaged_s"] >= args.min_dongle_engaged_s and d in per_dongle)
    print(f"loaded={len(npz_files)} skipped_short={n_skipped_short} skipped_empty={n_skipped_empty} "
          f"dongles_after_floor={len(dongles)}")

    if not dongles:
        print("No dongles passed the engaged-time floor.", file=sys.stderr)
        return 3

    # ---- per-dongle TSV ----
    per_dongle_tsv = out / "per_dongle.tsv"
    fields = ["dongle_id", "n_routes", "engaged_s", "accepted",
              "steer_ratio_mean", "stiffness_factor_mean", "lateral_delay_mean",
              "fingerprint", "vins", "branches",
              "has_curvature_params", "curv_use_frac_mean",
              "qfk_route_frac",
              "resid_td_mean_pose", "resid_td_rms_pose",
              "resid_td_mean_cs",   "resid_td_rms_cs",
              "resid_td_mean_qfk",  "resid_td_rms_qfk",
              "gain_pose", "gain_cs", "gain_qfk"]
    with open(per_dongle_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for d in dongles:
            arrs = per_dongle[d]
            meta = per_dongle_meta[d]
            n_route = max(meta["n_routes"], 1)
            row = {
                "dongle_id": d,
                "n_routes": meta["n_routes"],
                "engaged_s": round(meta["engaged_s"], 1),
                "accepted": meta["accepted"],
                "steer_ratio_mean": round(meta["steer_ratio_sum"] / max(meta["steer_ratio_n"], 1), 4)
                                     if meta["steer_ratio_n"] else "",
                "stiffness_factor_mean": round(meta["stiffness_factor_sum"] / max(meta["stiffness_factor_n"], 1), 4)
                                     if meta["stiffness_factor_n"] else "",
                "lateral_delay_mean": round(meta["lateral_delay_sum"] / max(meta["lateral_delay_n"], 1), 4)
                                     if meta["lateral_delay_n"] else "",
                "fingerprint": meta["car_fingerprint"],
                "vins": ",".join(sorted(meta["vins"]))[:60],
                "branches": ",".join(sorted(meta["branches"]))[:60],
                "has_curvature_params": int(meta["has_curvature_params"]),
                "curv_use_frac_mean": round(meta["curv_use_frac_sum"] / n_route, 3),
                "qfk_route_frac": round(meta["qfk_seen_routes"] / n_route, 3),
            }
            # Pooled residual per truth source.
            for ti, name in enumerate(TRUTH_NAMES):
                n = int(arrs["truth_n"][:, :, ti].sum())
                sx = float(arrs["resid_td_sum"][:, :, ti].sum())
                sxx = float(arrs["resid_td_sumsq"][:, :, ti].sum())
                if n > 0:
                    mean = sx / n
                    rms = math.sqrt(max(sxx / n, 0.0))
                else:
                    mean = rms = float("nan")
                row[f"resid_td_mean_{name.split('_')[0]}"] = round(mean, 7) if n > 0 else ""
                row[f"resid_td_rms_{name.split('_')[0]}"] = round(rms, 7) if n > 0 else ""
                # Pooled gain = mean(truth) / mean(desired) over samples where
                # this truth source was valid.  Use matching denominator:
                #   desired_sum_valid = truth_sum - resid_td_sum
                # (since resid_td_sum = sum(truth - desired) over valid samples).
                cur_edges = curvature_edges
                centers = (cur_edges[:-1] + cur_edges[1:]) / 2
                keep_c = np.where((np.abs(centers) > 5e-4) & (np.abs(centers) < 0.5))[0]
                t_sum = float(arrs["truth_sum"][:, keep_c, ti].sum())
                td_sum = float(arrs["resid_td_sum"][:, keep_c, ti].sum())
                d_sum_valid = t_sum - td_sum
                gain = t_sum / d_sum_valid if abs(d_sum_valid) > 1e-6 else float("nan")
                row[f"gain_{name.split('_')[0]}"] = round(gain, 4)
            w.writerow(row)
    print(f"  wrote {per_dongle_tsv}")

    # ---- per-dongle bucket NPZ ----
    out_bucket_path = out / "per_dongle_bucket.npz"
    bucket_payload = {
        "speed_edges": speed_edges,
        "curvature_edges": curvature_edges,
        "truth_names": np.array(TRUTH_NAMES, dtype=object),
        "dongles": np.array(dongles, dtype=object),
    }
    for k in keys_2d:
        bucket_payload[k] = np.stack([per_dongle[d][k] for d in dongles], axis=0)
    for k in keys_3d:
        bucket_payload[k] = np.stack([per_dongle[d][k] for d in dongles], axis=0)
    np.savez_compressed(out_bucket_path, **bucket_payload)
    print(f"  wrote {out_bucket_path}")

    # ---- cross-dongle TSV: per (speed, curvature) bucket, summarize across dongles ----
    cross_path = out / "cross_dongle.tsv"
    n_speed = len(speed_edges) - 1
    n_curv = len(curvature_edges) - 1
    fieldnames = ["speed_bin", "curvature_bin", "speed_lo", "speed_hi",
                  "curv_lo", "curv_hi"]
    for tn in TRUTH_NAMES:
        for k in ("n_dongles", "median_mean", "iqr_mean", "min_mean", "max_mean"):
            fieldnames.append(f"{tn}_{k}")
    with open(cross_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for s in range(n_speed):
            for c in range(n_curv):
                row = {
                    "speed_bin": s, "curvature_bin": c,
                    "speed_lo": round(float(speed_edges[s]), 2),
                    "speed_hi": round(float(speed_edges[s + 1]), 2),
                    "curv_lo": float(curvature_edges[c]),
                    "curv_hi": float(curvature_edges[c + 1]),
                }
                for ti, name in enumerate(TRUTH_NAMES):
                    per_dongle_mean = []
                    for d in dongles:
                        arrs = per_dongle[d]
                        n = int(arrs["truth_n"][s, c, ti])
                        if n < args.min_bucket_n:
                            continue
                        m = float(arrs["resid_td_sum"][s, c, ti]) / n
                        per_dongle_mean.append(m)
                    pdm = np.array(per_dongle_mean) if per_dongle_mean else np.array([])
                    row[f"{name}_n_dongles"] = len(pdm)
                    if len(pdm) > 0:
                        row[f"{name}_median_mean"] = float(np.median(pdm))
                        row[f"{name}_iqr_mean"] = float(np.percentile(pdm, 75) - np.percentile(pdm, 25)) if len(pdm) >= 2 else 0.0
                        row[f"{name}_min_mean"] = float(pdm.min())
                        row[f"{name}_max_mean"] = float(pdm.max())
                    else:
                        row[f"{name}_median_mean"] = row[f"{name}_iqr_mean"] = ""
                        row[f"{name}_min_mean"] = row[f"{name}_max_mean"] = ""
                w.writerow(row)
    print(f"  wrote {cross_path}")

    # ---- Human-readable summary ----
    summary_path = out / "fleet_summary.txt"
    lines = []
    lines.append(f"# Fleet lateral tracking summary")
    lines.append(f"# loaded={len(npz_files)} dongles_used={len(dongles)} "
                 f"skipped_short={n_skipped_short} skipped_empty={n_skipped_empty}")
    lines.append(f"# min_dongle_engaged_s={args.min_dongle_engaged_s}")
    lines.append("")
    total_engaged = sum(per_dongle_meta[d]["engaged_s"] for d in dongles)
    total_accepted = sum(per_dongle_meta[d]["accepted"] for d in dongles)
    lines.append(f"Total engaged seconds across fleet: {total_engaged:.0f} s")
    lines.append(f"Total accepted samples: {total_accepted}")
    lines.append("")
    lines.append("Per-dongle pooled residual (truth - desired) using pose_yaw truth:")
    lines.append(f"  {'dongle':17s}  {'n_routes':>9s}  {'engaged_s':>10s}  "
                 f"{'mean_resid':>12s}  {'rms_resid':>11s}  {'gain':>7s}  "
                 f"{'sr':>6s}  {'sf':>6s}  {'ld':>5s}  fingerprint")
    pose_means: list[tuple[str, float, float]] = []
    for d in dongles:
        arrs = per_dongle[d]
        meta = per_dongle_meta[d]
        ti = 0  # pose_yaw
        n = int(arrs["truth_n"][:, :, ti].sum())
        sx = float(arrs["resid_td_sum"][:, :, ti].sum())
        sxx = float(arrs["resid_td_sumsq"][:, :, ti].sum())
        mean = sx / n if n > 0 else float("nan")
        rms = math.sqrt(max(sxx / n, 0.0)) if n > 0 else float("nan")
        cur_edges = curvature_edges
        centers = (cur_edges[:-1] + cur_edges[1:]) / 2
        keep_c = np.where((np.abs(centers) > 5e-4) & (np.abs(centers) < 0.5))[0]
        t_sum = float(arrs["truth_sum"][:, keep_c, ti].sum())
        td_sum = float(arrs["resid_td_sum"][:, keep_c, ti].sum())
        d_sum_valid = t_sum - td_sum
        gain = t_sum / d_sum_valid if abs(d_sum_valid) > 1e-6 else float("nan")
        sr = meta["steer_ratio_sum"] / max(meta["steer_ratio_n"], 1) if meta["steer_ratio_n"] else float("nan")
        sf = meta["stiffness_factor_sum"] / max(meta["stiffness_factor_n"], 1) if meta["stiffness_factor_n"] else float("nan")
        ld = meta["lateral_delay_sum"] / max(meta["lateral_delay_n"], 1) if meta["lateral_delay_n"] else float("nan")
        lines.append(f"  {d:17s}  {meta['n_routes']:>9d}  {meta['engaged_s']:>10.0f}  "
                     f"{mean:>+12.7f}  {rms:>11.7f}  {gain:>7.3f}  "
                     f"{sr:>6.2f}  {sf:>6.2f}  {ld:>5.2f}  {meta['car_fingerprint']}")
        pose_means.append((d, mean, gain))
    lines.append("")
    if len(pose_means) >= 2:
        means = np.array([m for _, m, _ in pose_means])
        gains = np.array([g for _, _, g in pose_means])
        lines.append("Cross-dongle distribution (pose_yaw truth):")
        lines.append(f"  resid_mean: min={means.min():+.7f}  median={np.median(means):+.7f}  "
                     f"max={means.max():+.7f}   (range = {means.max()-means.min():.7f})")
        lines.append(f"  gain:       min={gains.min():.3f}  median={np.median(gains):.3f}  "
                     f"max={gains.max():.3f}   (range = {gains.max()-gains.min():.3f})")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {summary_path}")
    print()
    for ln in lines:
        print(ln)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
