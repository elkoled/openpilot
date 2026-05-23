#!/usr/bin/env python3
"""
Stage 4 — Cross-car aggregation and pre-registered hypothesis battery.

Reads per_segment.jsonl, aggregates by dongle (per-car) and then across the
fleet. Every population statistic carries N_cars contributing. Plots that
would be sourced from < MIN_CARS dongles are flagged in the output and the
report greys them out.

Pre-registered hypothesis tests (no peeking; tests defined before looking
at the data):

  1. Population residual heatmap on the learner 7x12 grid + extended 7x16.
  2. Fraction of residual mass inside vs outside the learner-supported region.
  3. Per-car gain (out-of-bounds excluded), lag, deadband, left/right asymmetry,
     transient overshoot proxy.
  4. Cross-car Spearman correlation between residual magnitude and:
     steerRatio, stiffnessFactor, lateralDelay, build_year, engaged distance,
     EPS power, lat_accel_p95.
  5. Learner-replay vs batch-fit per-bucket agreement, per car.
  6. PID-on vs PID-off engaged-route residual distribution comparison.

The decision tree is applied in report.py, not here. This file only emits
numbers and confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from openpilot.tools.vw_id4_lateral.grid import (
  BucketAccumulator, EXTENDED_SHAPE, LEARNER_SHAPE,
  EXTENDED_CURVATURE_EDGES, EXTENDED_SPEED_ANCHORS,
  LEARNER_CURVATURE_EDGES, LEARNER_SPEED_ANCHORS,
)

MIN_CARS_PER_BUCKET = 5
MIN_CARS_PER_PLOT = 5
MIN_BUCKET_SAMPLES_PER_CAR = 30  # ignore a car's contribution to a bucket below this


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
  """Spearman rank correlation. Returns (rho, n_valid). Falls back to NaN
  when fewer than 5 valid pairs."""
  mask = np.isfinite(x) & np.isfinite(y)
  n = int(np.sum(mask))
  if n < 5:
    return float("nan"), n
  rx = np.argsort(np.argsort(x[mask]))
  ry = np.argsort(np.argsort(y[mask]))
  return float(np.corrcoef(rx, ry)[0, 1]), n


def _bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 200,
                  ci: float = 0.9) -> tuple[float, float, float]:
  """Bootstrap median + (lo, hi) for the given confidence interval."""
  values = values[np.isfinite(values)]
  if len(values) == 0:
    return float("nan"), float("nan"), float("nan")
  rng = np.random.default_rng(42)
  boots = np.empty(n_boot, dtype=np.float64)
  for i in range(n_boot):
    sample = rng.choice(values, size=len(values), replace=True)
    boots[i] = stat(sample)
  alpha = (1.0 - ci) / 2
  return float(stat(values)), float(np.percentile(boots, 100 * alpha)), \
         float(np.percentile(boots, 100 * (1.0 - alpha)))


def per_car_aggregate(records: list[dict]) -> dict[str, dict]:
  """Pool segment-level bucket accumulators per dongle. Also collects per-car
  scalar features (medians of segment scalars) and the learner-vs-batch fit."""
  by_car: dict[str, dict] = defaultdict(lambda: {
    "segments": 0,
    "engaged_s": 0.0,
    "ok_segments": 0,
    "vin": "",
    "build_year": "",
    "has_pid": False,
    "buckets_learner_gated": BucketAccumulator(LEARNER_SHAPE),
    "buckets_learner_ungated": BucketAccumulator(LEARNER_SHAPE),
    "buckets_extended_ungated": BucketAccumulator(EXTENDED_SHAPE),
    "highway_residual_p50_list": [],
    "highway_residual_p95_list": [],
    "steer_ratio_list": [],
    "stiffness_list": [],
    "lateral_delay_list": [],
    "eps_power_list": [],
    "per_speed_gain": defaultdict(list),
    "learner_replay_bias_sum": np.zeros(LEARNER_SHAPE, dtype=np.float64),
    "learner_replay_bias_w_sum": np.zeros(LEARNER_SHAPE, dtype=np.float64),
    "batch_fit_bias_sum": np.zeros(LEARNER_SHAPE, dtype=np.float64),
    "batch_fit_bias_w_sum": np.zeros(LEARNER_SHAPE, dtype=np.float64),
  })

  for r in records:
    dongle = r.get("dongle") or r.get("route", "").split("/")[0]
    by_car[dongle]["segments"] += 1
    if r.get("ok"):
      by_car[dongle]["ok_segments"] += 1
    by_car[dongle]["engaged_s"] += float(r.get("engaged_s", 0.0))
    if r.get("vin"):
      by_car[dongle]["vin"] = r["vin"]
    if r.get("build_year"):
      by_car[dongle]["build_year"] = r["build_year"]
    if r.get("has_live_curvature_parameters"):
      by_car[dongle]["has_pid"] = True

    for key, shape in (("buckets_learner_gated", LEARNER_SHAPE),
                       ("buckets_learner_ungated", LEARNER_SHAPE),
                       ("buckets_extended_ungated", EXTENDED_SHAPE)):
      bd = r.get(key)
      if bd:
        try:
          by_car[dongle][key] += BucketAccumulator.from_dict(bd)
        except Exception:
          pass

    for k, lst_key in (("highway_abs_residual_p50", "highway_residual_p50_list"),
                       ("highway_abs_residual_p95", "highway_residual_p95_list"),
                       ("mean_steer_ratio", "steer_ratio_list"),
                       ("mean_stiffness_factor", "stiffness_list"),
                       ("mean_lateral_delay", "lateral_delay_list"),
                       ("mean_eps_power", "eps_power_list")):
      val = r.get(k)
      if val is not None and math.isfinite(val):
        by_car[dongle][lst_key].append(float(val))

    for k, v in (r.get("per_speed_bin_gain") or {}).items():
      n = (r.get("per_speed_bin_gain_n") or {}).get(k, 1)
      by_car[dongle]["per_speed_gain"][k].append((float(v), int(n)))

    lr = r.get("learner_replay") or {}
    bf = r.get("batch_fit") or {}
    if lr.get("bias") is not None and bf.get("bias") is not None:
      lr_bias = np.array(lr["bias"], dtype=np.float64)
      lr_counts = np.array(lr["counts"], dtype=np.float64)
      bf_bias = np.array(bf["bias"], dtype=np.float64)
      bf_counts = np.array(bf["counts"], dtype=np.float64)
      # weight by sample count
      by_car[dongle]["learner_replay_bias_sum"] += lr_bias * lr_counts
      by_car[dongle]["learner_replay_bias_w_sum"] += lr_counts
      by_car[dongle]["batch_fit_bias_sum"] += bf_bias * bf_counts
      by_car[dongle]["batch_fit_bias_w_sum"] += bf_counts

  return dict(by_car)


def car_bucket_bias(acc: BucketAccumulator) -> tuple[np.ndarray, np.ndarray]:
  """Returns (mean_signed_error, count) per bucket."""
  with np.errstate(invalid="ignore", divide="ignore"):
    mean = np.where(acc.count > 0, acc.sum_err / np.maximum(acc.count, 1), 0.0)
  return mean.astype(np.float64), acc.count.astype(np.float64)


def population_bucket_stats(per_car: dict[str, dict],
                            bucket_key: str,
                            shape: tuple) -> dict:
  """Per-bucket population stats across cars. For each bucket, we collect one
  scalar per car (their weighted mean signed error in that bucket) and report
  median, IQR, N_cars. Cars contributing fewer than MIN_BUCKET_SAMPLES_PER_CAR
  to a bucket are excluded from THAT bucket only."""
  S, C = shape
  per_bucket_values: list[list[float]] = [[] for _ in range(S * C)]
  per_bucket_n_cars = np.zeros((S, C), dtype=np.int64)
  for car, cd in per_car.items():
    bias, counts = car_bucket_bias(cd[bucket_key])
    for s in range(S):
      for c in range(C):
        if counts[s, c] >= MIN_BUCKET_SAMPLES_PER_CAR:
          per_bucket_values[s * C + c].append(float(bias[s, c]))
          per_bucket_n_cars[s, c] += 1
  median = np.full((S, C), np.nan)
  q25 = np.full((S, C), np.nan)
  q75 = np.full((S, C), np.nan)
  for s in range(S):
    for c in range(C):
      vals = np.array(per_bucket_values[s * C + c], dtype=np.float64)
      if len(vals) >= MIN_CARS_PER_BUCKET:
        median[s, c] = float(np.median(vals))
        q25[s, c] = float(np.percentile(vals, 25))
        q75[s, c] = float(np.percentile(vals, 75))
  return {
    "median": median.tolist(),
    "q25": q25.tolist(),
    "q75": q75.tolist(),
    "n_cars": per_bucket_n_cars.tolist(),
    "shape": list(shape),
  }


def hypothesis_battery(per_car: dict[str, dict]) -> dict:
  """Per-car summary scalars for the hypothesis tests. Returns a dict of
  numpy arrays keyed by hypothesis name."""
  cars = list(per_car.keys())
  result = {
    "cars": cars,
    "engaged_s": np.array([per_car[c]["engaged_s"] for c in cars], dtype=np.float64),
    "build_year": [per_car[c]["build_year"] for c in cars],
    "has_pid": np.array([per_car[c]["has_pid"] for c in cars], dtype=bool),
  }
  # H1: highway residual p50 / p95
  result["highway_p50"] = np.array(
    [np.median(per_car[c]["highway_residual_p50_list"]) if per_car[c]["highway_residual_p50_list"]
     else np.nan for c in cars], dtype=np.float64)
  result["highway_p95"] = np.array(
    [np.median(per_car[c]["highway_residual_p95_list"]) if per_car[c]["highway_residual_p95_list"]
     else np.nan for c in cars], dtype=np.float64)
  # H2: per-speed gain (median across segments per car per speed)
  for kph in (40, 60, 80, 100, 120):
    key = f"{kph}kph"
    vals = []
    for c in cars:
      pairs = per_car[c]["per_speed_gain"].get(key, [])
      if not pairs:
        vals.append(np.nan)
        continue
      ws = np.array([n for _, n in pairs], dtype=np.float64)
      gs = np.array([g for g, _ in pairs], dtype=np.float64)
      vals.append(float(np.average(gs, weights=ws)))
    result[f"gain_{key}"] = np.array(vals, dtype=np.float64)
  # H3: live-params summaries per car
  for k, lst_key in (("steer_ratio", "steer_ratio_list"),
                     ("stiffness", "stiffness_list"),
                     ("lateral_delay", "lateral_delay_list"),
                     ("eps_power", "eps_power_list")):
    result[k] = np.array(
      [np.median(per_car[c][lst_key]) if per_car[c][lst_key] else np.nan
       for c in cars], dtype=np.float64)
  # H4: learner-vs-batch agreement per car (Pearson on flattened weighted-mean
  # biases, restricted to buckets with > MIN_BUCKET_SAMPLES_PER_CAR samples).
  lr_vs_bf = []
  for c in cars:
    cd = per_car[c]
    lw = cd["learner_replay_bias_w_sum"]
    bw = cd["batch_fit_bias_w_sum"]
    with np.errstate(invalid="ignore", divide="ignore"):
      lb = np.where(lw > 0, cd["learner_replay_bias_sum"] / np.maximum(lw, 1), np.nan)
      bb = np.where(bw > 0, cd["batch_fit_bias_sum"] / np.maximum(bw, 1), np.nan)
    mask = (lw > MIN_BUCKET_SAMPLES_PER_CAR) & (bw > MIN_BUCKET_SAMPLES_PER_CAR) \
           & np.isfinite(lb) & np.isfinite(bb)
    if int(np.sum(mask)) < 6:
      lr_vs_bf.append(np.nan)
      continue
    lr_vs_bf.append(float(np.corrcoef(lb[mask], bb[mask])[0, 1]))
  result["learner_vs_batch_pearson"] = np.array(lr_vs_bf, dtype=np.float64)
  # H5: left/right asymmetry — per-car summed |signed_err_left - signed_err_right|
  # in the inner (low-curvature) region.
  inner_asym = []
  for c in cars:
    acc = per_car[c]["buckets_learner_gated"]
    # inner half of the bucket range (low curvature, where the dynamic_steering
    # learner says the residual matters)
    inner_c = LEARNER_SHAPE[1] // 2
    lt = acc.sum_err_left[:, :inner_c]
    rt = acc.sum_err_right[:, :inner_c]
    nl = acc.cnt_left[:, :inner_c]
    nr = acc.cnt_right[:, :inner_c]
    with np.errstate(invalid="ignore", divide="ignore"):
      ml = np.where(nl > 5, lt / np.maximum(nl, 1), np.nan)
      mr = np.where(nr > 5, rt / np.maximum(nr, 1), np.nan)
    diff = ml - mr
    if np.sum(np.isfinite(diff)) < 3:
      inner_asym.append(np.nan)
    else:
      inner_asym.append(float(np.nanmedian(diff)))
  result["inner_left_right_asym"] = np.array(inner_asym, dtype=np.float64)
  return result


def inside_vs_outside_grid_mass(per_car: dict[str, dict]) -> dict:
  """For each car, fraction of |signed residual| × count that lands inside
  vs outside the learner-supported region of the extended grid.

  The learner-supported region is the inner S x C cells of the extended grid
  (the first 7 speed rows × first 12 curvature columns)."""
  S_ext, C_ext = EXTENDED_SHAPE
  S_in, C_in = LEARNER_SHAPE
  fractions = {}
  for car, cd in per_car.items():
    acc = cd["buckets_extended_ungated"]
    abs_mass = np.abs(acc.sum_err)  # |signed| at the population level
    inside = abs_mass[:S_in, :C_in].sum()
    total = abs_mass.sum()
    if total <= 0:
      fractions[car] = float("nan")
    else:
      fractions[car] = float(inside / total)
  return fractions


def pid_vs_nopid_residual_comparison(per_car: dict[str, dict]) -> dict:
  """Compare highway_p95 distributions between PID-on and PID-off dongles.
  Uses median + bootstrap CI. Returns counts so the caller can warn on N<5."""
  pid_on = []
  pid_off = []
  for cd in per_car.values():
    val = np.median(cd["highway_residual_p95_list"]) if cd["highway_residual_p95_list"] else np.nan
    if not math.isfinite(val):
      continue
    if cd["has_pid"]:
      pid_on.append(val)
    else:
      pid_off.append(val)
  pid_on = np.array(pid_on, dtype=np.float64)
  pid_off = np.array(pid_off, dtype=np.float64)
  on_median, on_lo, on_hi = _bootstrap_ci(pid_on)
  off_median, off_lo, off_hi = _bootstrap_ci(pid_off)
  return {
    "pid_on": {"n_cars": int(len(pid_on)), "median": on_median,
               "ci_lo": on_lo, "ci_hi": on_hi, "values": pid_on.tolist()},
    "pid_off": {"n_cars": int(len(pid_off)), "median": off_median,
                "ci_lo": off_lo, "ci_hi": off_hi, "values": pid_off.tolist()},
  }


def covariate_correlations(hyp: dict) -> dict:
  """Spearman ρ between per-car residual magnitude (highway_p95) and each
  candidate covariate, with sample size."""
  y = hyp["highway_p95"]
  covariates = {
    "steer_ratio": hyp["steer_ratio"],
    "stiffness": hyp["stiffness"],
    "lateral_delay": hyp["lateral_delay"],
    "eps_power": hyp["eps_power"],
    "engaged_s": hyp["engaged_s"],
    "gain_80kph": hyp.get("gain_80kph", np.array([])),
    "gain_100kph": hyp.get("gain_100kph", np.array([])),
    "gain_120kph": hyp.get("gain_120kph", np.array([])),
  }
  by_var = hyp.get("build_year", [])
  if by_var:
    try:
      by_arr = np.array([float(b) if b else np.nan for b in by_var], dtype=np.float64)
      covariates["build_year"] = by_arr
    except Exception:
      pass

  out = {}
  for k, v in covariates.items():
    if len(v) != len(y) or len(v) == 0:
      out[k] = {"rho": float("nan"), "n": 0}
      continue
    rho, n = _spearman(v, y)
    out[k] = {"rho": rho, "n": n}
  return out


def aggregate_fleet(per_segment_path: str) -> dict:
  records = []
  with open(per_segment_path) as f:
    for line in f:
      try:
        records.append(json.loads(line))
      except Exception:
        continue
  print(f"[analyze] loaded {len(records)} per-segment records", flush=True)

  per_car = per_car_aggregate(records)
  n_cars = len(per_car)
  pop_learner_gated = population_bucket_stats(per_car, "buckets_learner_gated", LEARNER_SHAPE)
  pop_learner_ungated = population_bucket_stats(per_car, "buckets_learner_ungated", LEARNER_SHAPE)
  pop_extended = population_bucket_stats(per_car, "buckets_extended_ungated", EXTENDED_SHAPE)
  hyp = hypothesis_battery(per_car)
  inside_mass = inside_vs_outside_grid_mass(per_car)
  pid_cmp = pid_vs_nopid_residual_comparison(per_car)
  covar = covariate_correlations(hyp)

  # Per-car compact summaries (avoid pickling the full BucketAccumulators back)
  cars_summary = []
  for car, cd in per_car.items():
    cars_summary.append({
      "dongle": car,
      "segments": cd["segments"],
      "ok_segments": cd["ok_segments"],
      "engaged_s": cd["engaged_s"],
      "vin": cd["vin"],
      "build_year": cd["build_year"],
      "has_pid": cd["has_pid"],
      "highway_p50_median": float(np.median(cd["highway_residual_p50_list"]))
        if cd["highway_residual_p50_list"] else float("nan"),
      "highway_p95_median": float(np.median(cd["highway_residual_p95_list"]))
        if cd["highway_residual_p95_list"] else float("nan"),
    })

  out = {
    "n_cars": n_cars,
    "n_segments": len(records),
    "pop_learner_gated": pop_learner_gated,
    "pop_learner_ungated": pop_learner_ungated,
    "pop_extended_ungated": pop_extended,
    "hypothesis_battery": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in hyp.items()},
    "inside_mass_per_car": inside_mass,
    "pid_vs_nopid": pid_cmp,
    "covariate_spearman": covar,
    "cars": cars_summary,
    "min_cars_per_bucket": MIN_CARS_PER_BUCKET,
    "min_cars_per_plot": MIN_CARS_PER_PLOT,
    "min_bucket_samples_per_car": MIN_BUCKET_SAMPLES_PER_CAR,
  }
  # Also dump per-car bucket bias matrices for the report (per-car heatmap appendix)
  per_car_bias = {}
  for car, cd in per_car.items():
    bias, counts = car_bucket_bias(cd["buckets_learner_gated"])
    per_car_bias[car] = {
      "bias": bias.tolist(),
      "counts": counts.tolist(),
    }
  out["per_car_learner_gated_bias"] = per_car_bias
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--in", dest="in_path", default="per_segment.jsonl")
  p.add_argument("--out", default="fleet_stats.pkl")
  args = p.parse_args()
  stats = aggregate_fleet(args.in_path)
  with open(args.out, "wb") as f:
    pickle.dump(stats, f)
  print(f"[analyze] wrote {args.out}  n_cars={stats['n_cars']}  "
        f"n_segments={stats['n_segments']}", flush=True)


if __name__ == "__main__":
  sys.exit(main())
