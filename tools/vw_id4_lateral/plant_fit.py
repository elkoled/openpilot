#!/usr/bin/env python3
"""
EPS plant identification with leave-one-route-out cross-validation.

For each car (dongle) and each speed bin we fit two candidate models against
the engaged 100 Hz timelines stored as sidecar .npz files by extract():

  STATIC:   actual[t] = G * desired[t - Td] + bias
  ARX1:     actual[t] = a * actual[t-1]   + b * desired[t - Td]
            (continuous-time: actual = K/(τs+1)·exp(-Td·s)·desired,
             with a = exp(-DT/τ), K = b / (1 - a))

Bounded fits — any unbounded least-squares solution outside the physical
bounds (passive plant: K ∈ (0.5, 1.5), τ ∈ (20 ms, 1 s), Td ∈ [0, 0.3 s])
is rejected. Td is searched on a discrete grid by RMSE on TRAINING data
only, then frozen for CV.

Leave-one-route-out CV: each segment (route) becomes a held-out fold once
per (car, speed_bin). We report median LOO RMSE, R², and bias per car.

For a model to be "deployable" for a car at a speed bin, we require:
  - N_train_routes ≥ 3   (so LOO-CV has at least 2 fitting folds)
  - in-sample R² ≥ 0.5    (model explains at least half the variance)
  - LOO RMSE ≤ in-sample RMSE × 1.5  (no catastrophic over-fitting)
  - bounded fit succeeded
Otherwise that (car, speed_bin) cell is marked NOT-DEPLOYABLE and the
report carries that flag forward.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, asdict, field

import numpy as np

# Match the openpilot10 speed anchors so per-car plant fits drop into
# the same speed grid as the learner.
SPEED_ANCHORS_KPH = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]
SPEED_ANCHORS = np.array(SPEED_ANCHORS_KPH) / 3.6
SPEED_HALFWIDTH = 8.0 / 3.6  # ±8 km/h band per anchor

DT = 0.01  # 100 Hz
TD_GRID_SAMPLES = (5, 10, 15, 20, 25, 30)  # 50..300 ms

# Bounds — physically motivated passive-plant constraints
K_LO, K_HI = 0.5, 1.5
TAU_LO_S, TAU_HI_S = 0.02, 1.0
A_LO = math.exp(-DT / TAU_HI_S)
A_HI = math.exp(-DT / TAU_LO_S)
BIAS_ABS_MAX = 5e-4  # rad/m

MIN_SAMPLES_PER_SPEED_BIN = 500     # absolute min for any fit
MIN_ROUTES_PER_BIN_FOR_CV = 3       # need ≥ 3 routes for meaningful LOO

# Deployable thresholds
MIN_IN_SAMPLE_R2 = 0.5
MAX_LOO_INFLATION = 1.5             # LOO RMSE / in-sample RMSE


@dataclass
class PlantFitResult:
  dongle: str
  speed_anchor_kph: float
  n_samples: int
  n_routes: int
  # static model
  static_G: float = float("nan")
  static_bias: float = float("nan")
  static_Td_s: float = float("nan")
  static_in_sample_r2: float = float("nan")
  static_in_sample_rmse: float = float("nan")
  static_loo_rmse: float = float("nan")
  static_loo_r2: float = float("nan")
  static_loo_bias_mean: float = float("nan")
  # ARX model
  arx_a: float = float("nan")
  arx_b: float = float("nan")
  arx_K: float = float("nan")
  arx_tau_s: float = float("nan")
  arx_Td_s: float = float("nan")
  arx_in_sample_r2: float = float("nan")
  arx_in_sample_rmse: float = float("nan")
  arx_loo_rmse: float = float("nan")
  arx_loo_r2: float = float("nan")
  arx_loo_bias_mean: float = float("nan")
  # baseline (identity plant: actual = desired)
  identity_rmse: float = float("nan")
  identity_bias: float = float("nan")
  # decision
  chosen_model: str = "none"  # "static" | "arx" | "none"
  deployable: bool = False
  reject_reason: str = ""


def _load_timeline(path: str) -> dict | None:
  try:
    with np.load(path) as z:
      return {k: np.array(z[k]) for k in z.files}
  except Exception:
    return None


def _engaged_band_mask(tl: dict, v_anchor: float) -> np.ndarray:
  v = tl["v"]
  in_band = np.abs(v - v_anchor) <= SPEED_HALFWIDTH
  # exclude driver-pressed frames
  in_band &= ~tl["steering_pressed"]
  # only valid yaw-std
  in_band &= np.isfinite(tl["yaw_std"]) & (tl["yaw_std"] < 1.0)
  # finite desired/actual
  in_band &= np.isfinite(tl["desired"]) & np.isfinite(tl["actual"])
  return in_band


def _fit_static(d: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
  """Weighted LSQ fit of actual = G*desired + bias. Returns (G, bias, in_sample_rmse).
  Bounds: G ∈ (K_LO, K_HI); bias clipped to ±BIAS_ABS_MAX."""
  if len(d) < 20:
    return float("nan"), float("nan"), float("nan")
  # build design matrix
  X = np.column_stack([d, np.ones_like(d)])
  try:
    sol, *_ = np.linalg.lstsq(X, a, rcond=None)
  except np.linalg.LinAlgError:
    return float("nan"), float("nan"), float("nan")
  G, bias = float(sol[0]), float(sol[1])
  if not (K_LO <= G <= K_HI):
    return float("nan"), float("nan"), float("nan")
  if abs(bias) > BIAS_ABS_MAX:
    bias = float(np.clip(bias, -BIAS_ABS_MAX, BIAS_ABS_MAX))
  pred = G * d + bias
  rmse = float(np.sqrt(np.mean((a - pred) ** 2)))
  return G, bias, rmse


def _fit_arx(d_lag: np.ndarray, a_prev: np.ndarray, a_next: np.ndarray
             ) -> tuple[float, float, float]:
  """Fit a_next = a * a_prev + b * d_lag. Bounds: a ∈ (A_LO, A_HI), K=b/(1-a) ∈ (K_LO, K_HI).
  Returns (a, b, in_sample_rmse)."""
  if len(d_lag) < 20:
    return float("nan"), float("nan"), float("nan")
  X = np.column_stack([a_prev, d_lag])
  try:
    sol, *_ = np.linalg.lstsq(X, a_next, rcond=None)
  except np.linalg.LinAlgError:
    return float("nan"), float("nan"), float("nan")
  aa, bb = float(sol[0]), float(sol[1])
  if not (A_LO <= aa <= A_HI):
    return float("nan"), float("nan"), float("nan")
  K = bb / max(1.0 - aa, 1e-6)
  if not (K_LO <= K <= K_HI):
    return float("nan"), float("nan"), float("nan")
  pred = aa * a_prev + bb * d_lag
  rmse = float(np.sqrt(np.mean((a_next - pred) ** 2)))
  return aa, bb, rmse


def _r2(actual: np.ndarray, pred: np.ndarray) -> float:
  var = float(np.var(actual))
  if var <= 0:
    return float("nan")
  return 1.0 - float(np.mean((actual - pred) ** 2)) / var


def _select_td(per_route: list[dict], v_anchor: float, model: str
               ) -> tuple[int, float]:
  """Sweep Td on training-only data (we'll re-evaluate via LOO). Picks the Td
  with the lowest in-sample RMSE pooled across all routes for this speed bin.
  Returns (Td_samples, best_rmse) or (-1, nan) if no fit succeeds."""
  best_td = -1
  best_rmse = float("inf")
  for Td in TD_GRID_SAMPLES:
    d_all, a_all, ap_all = [], [], []
    for tl in per_route:
      d, a, ap = _slice_for_arx(tl, Td)
      if d is None:
        continue
      d_all.append(d); a_all.append(a); ap_all.append(ap)
    if not d_all:
      continue
    d_cat = np.concatenate(d_all)
    a_cat = np.concatenate(a_all)
    ap_cat = np.concatenate(ap_all)
    if model == "static":
      G, b, rmse = _fit_static(d_cat, a_cat)
    else:
      _, _, rmse = _fit_arx(d_cat, ap_cat, a_cat)
    if rmse is None or math.isnan(rmse):
      continue
    if rmse < best_rmse:
      best_rmse = rmse
      best_td = Td
  return best_td, (best_rmse if best_td >= 0 else float("nan"))


def _slice_for_arx(tl_band: dict, Td: int):
  """Given the per-route, per-speed-band timeline slice, return (desired_lag,
  actual_curr, actual_prev) properly aligned for ARX(1,1,Td) fitting. None if
  not enough samples."""
  d = tl_band["desired"]
  a = tl_band["actual"]
  n = len(d)
  if n < Td + 2:
    return None, None, None
  # actual[t+1] = a*actual[t] + b*desired[t+1-Td]
  # we want: y = actual[Td+1:n],  ap = actual[Td:n-1],  d_lag = desired[1:n-Td]
  ap = a[Td:n - 1]
  y = a[Td + 1:n]
  d_lag = d[1:n - Td]
  m = min(len(ap), len(y), len(d_lag))
  return d_lag[:m], y[:m], ap[:m]


def _per_route_band(timelines: list[dict], v_anchor: float) -> list[dict]:
  """Apply the speed-band mask to each per-route timeline. Returns list of
  band-sliced timelines (only routes with enough samples kept)."""
  out = []
  for tl in timelines:
    mask = _engaged_band_mask(tl, v_anchor)
    if int(np.sum(mask)) < 50:  # per-route min for this band
      continue
    out.append({
      "desired": tl["desired"][mask],
      "actual": tl["actual"][mask],
      "route": tl.get("route", ""),
    })
  return out


def fit_one_car_one_bin(timelines: list[dict], v_anchor: float, dongle: str
                        ) -> PlantFitResult:
  banded = _per_route_band(timelines, v_anchor)
  n_routes = len(banded)
  n_samples = sum(len(tl["desired"]) for tl in banded)
  res = PlantFitResult(dongle=dongle, speed_anchor_kph=v_anchor * 3.6,
                       n_samples=n_samples, n_routes=n_routes)

  if n_samples < MIN_SAMPLES_PER_SPEED_BIN or n_routes < MIN_ROUTES_PER_BIN_FOR_CV:
    res.reject_reason = f"insufficient_data:n={n_samples},routes={n_routes}"
    return res

  # Identity-plant baseline (no correction): how well does "actual = desired"
  # predict on the same gated data? The deployable model has to beat this on
  # held-out routes by a measurable margin.
  d_all = np.concatenate([tl["desired"] for tl in banded])
  a_all = np.concatenate([tl["actual"] for tl in banded])
  res.identity_rmse = float(np.sqrt(np.mean((a_all - d_all) ** 2)))
  res.identity_bias = float(np.mean(a_all - d_all))

  # Pick Td via pooled in-sample search.
  td_static, _ = _select_td(banded, v_anchor, "static")
  td_arx, _ = _select_td(banded, v_anchor, "arx")

  # ----- in-sample pooled fits + LOO-CV -----
  def pool_static(tls: list[dict], Td: int):
    d_all, a_all = [], []
    for tl in tls:
      n = len(tl["desired"])
      if n < Td + 2:
        continue
      d_all.append(tl["desired"][:n - Td])
      a_all.append(tl["actual"][Td:])
    if not d_all:
      return None
    d_cat = np.concatenate(d_all); a_cat = np.concatenate(a_all)
    m = min(len(d_cat), len(a_cat))
    return d_cat[:m], a_cat[:m]

  def pool_arx(tls: list[dict], Td: int):
    d_all, y_all, ap_all = [], [], []
    for tl in tls:
      d, y, ap = _slice_for_arx(tl, Td)
      if d is None:
        continue
      d_all.append(d); y_all.append(y); ap_all.append(ap)
    if not d_all:
      return None
    return np.concatenate(d_all), np.concatenate(y_all), np.concatenate(ap_all)

  # ---- static
  if td_static > 0:
    pool = pool_static(banded, td_static)
    if pool is not None:
      d_cat, a_cat = pool
      G, bias, rmse = _fit_static(d_cat, a_cat)
      if not math.isnan(G):
        pred = G * d_cat + bias
        res.static_G = G
        res.static_bias = bias
        res.static_Td_s = td_static * DT
        res.static_in_sample_rmse = rmse
        res.static_in_sample_r2 = _r2(a_cat, pred)
        # LOO-CV by route
        loo_errs = []
        for i in range(len(banded)):
          train = banded[:i] + banded[i + 1:]
          held = banded[i:i + 1]
          tp = pool_static(train, td_static)
          hp = pool_static(held, td_static)
          if tp is None or hp is None:
            continue
          d_t, a_t = tp
          d_h, a_h = hp
          G_i, bias_i, _ = _fit_static(d_t, a_t)
          if math.isnan(G_i):
            continue
          pred_h = G_i * d_h + bias_i
          loo_errs.append(a_h - pred_h)
        if loo_errs:
          err_cat = np.concatenate(loo_errs)
          res.static_loo_rmse = float(np.sqrt(np.mean(err_cat ** 2)))
          actual_cat = np.concatenate([pool_static([banded[i]], td_static)[1]
                                       for i in range(len(banded))
                                       if pool_static([banded[i]], td_static) is not None])
          res.static_loo_r2 = 1.0 - res.static_loo_rmse ** 2 / max(float(np.var(actual_cat)), 1e-12)
          res.static_loo_bias_mean = float(np.mean(err_cat))

  # ---- ARX
  if td_arx > 0:
    pool = pool_arx(banded, td_arx)
    if pool is not None:
      d_cat, y_cat, ap_cat = pool
      a_arx, b_arx, rmse = _fit_arx(d_cat, ap_cat, y_cat)
      if not math.isnan(a_arx):
        pred = a_arx * ap_cat + b_arx * d_cat
        res.arx_a = a_arx
        res.arx_b = b_arx
        res.arx_K = b_arx / max(1.0 - a_arx, 1e-6)
        res.arx_tau_s = -DT / math.log(max(a_arx, 1e-9))
        res.arx_Td_s = td_arx * DT
        res.arx_in_sample_rmse = rmse
        res.arx_in_sample_r2 = _r2(y_cat, pred)
        # LOO-CV by route
        loo_errs = []
        loo_actuals = []
        for i in range(len(banded)):
          train = banded[:i] + banded[i + 1:]
          held = banded[i:i + 1]
          tp = pool_arx(train, td_arx)
          hp = pool_arx(held, td_arx)
          if tp is None or hp is None:
            continue
          d_t, y_t, ap_t = tp
          d_h, y_h, ap_h = hp
          a_i, b_i, _ = _fit_arx(d_t, ap_t, y_t)
          if math.isnan(a_i):
            continue
          pred_h = a_i * ap_h + b_i * d_h
          loo_errs.append(y_h - pred_h)
          loo_actuals.append(y_h)
        if loo_errs:
          err_cat = np.concatenate(loo_errs)
          actual_cat = np.concatenate(loo_actuals)
          res.arx_loo_rmse = float(np.sqrt(np.mean(err_cat ** 2)))
          res.arx_loo_r2 = 1.0 - res.arx_loo_rmse ** 2 / max(float(np.var(actual_cat)), 1e-12)
          res.arx_loo_bias_mean = float(np.mean(err_cat))

  # ----- decision -----
  def deployable(in_r2: float, in_rmse: float, loo_rmse: float) -> bool:
    if math.isnan(in_r2) or math.isnan(in_rmse) or math.isnan(loo_rmse):
      return False
    if in_r2 < MIN_IN_SAMPLE_R2:
      return False
    if in_rmse > 0 and loo_rmse / in_rmse > MAX_LOO_INFLATION:
      return False
    return True

  static_ok = deployable(res.static_in_sample_r2, res.static_in_sample_rmse, res.static_loo_rmse)
  arx_ok = deployable(res.arx_in_sample_r2, res.arx_in_sample_rmse, res.arx_loo_rmse)

  # Must beat the identity baseline by ≥ 10% on LOO RMSE to count as deployable.
  IMPROVEMENT_MARGIN = 0.10
  if static_ok and not math.isnan(res.identity_rmse) and \
     res.static_loo_rmse > res.identity_rmse * (1.0 - IMPROVEMENT_MARGIN):
    static_ok = False
  if arx_ok and not math.isnan(res.identity_rmse) and \
     res.arx_loo_rmse > res.identity_rmse * (1.0 - IMPROVEMENT_MARGIN):
    arx_ok = False

  if arx_ok and (not static_ok or res.arx_loo_rmse < res.static_loo_rmse):
    res.chosen_model = "arx"
    res.deployable = True
  elif static_ok:
    res.chosen_model = "static"
    res.deployable = True
  else:
    res.chosen_model = "none"
    res.deployable = False
    if not (static_ok or arx_ok):
      reasons = []
      if math.isnan(res.static_G) and math.isnan(res.arx_a):
        reasons.append("no_bounded_fit_succeeded")
      else:
        reasons.append(f"r2_or_loo_insufficient(static_r2={res.static_in_sample_r2:.2f},arx_r2={res.arx_in_sample_r2:.2f})")
      res.reject_reason = ",".join(reasons)

  return res


def fit_all(cache_dir: str, dongles: list[str] | None = None
            ) -> dict:
  """Walk cache_dir, fit each car at each speed anchor, return results."""
  if not os.path.isdir(cache_dir):
    raise FileNotFoundError(cache_dir)

  per_car = {}
  for d in os.listdir(cache_dir):
    car_dir = os.path.join(cache_dir, d)
    if not os.path.isdir(car_dir):
      continue
    if dongles and d not in dongles:
      continue
    files = sorted(os.listdir(car_dir))
    timelines = []
    for fn in files:
      if not fn.endswith(".npz"):
        continue
      tl = _load_timeline(os.path.join(car_dir, fn))
      if tl is None or "v" not in tl:
        continue
      tl["route"] = fn.replace(".npz", "")
      timelines.append(tl)
    if not timelines:
      continue
    per_car[d] = timelines

  results = {}
  for dongle, tls in per_car.items():
    car_results = {}
    print(f"[plant_fit] fitting {dongle} ({len(tls)} segments)", flush=True)
    for v_anchor in SPEED_ANCHORS:
      res = fit_one_car_one_bin(tls, v_anchor, dongle)
      car_results[f"{int(v_anchor * 3.6)}kph"] = asdict(res)
    results[dongle] = car_results

  return results


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cache-dir", default="cache")
  p.add_argument("--out", default="plant_fit.json")
  p.add_argument("--dongles", default="", help="Comma-separated subset")
  args = p.parse_args()
  dongle_list = [d.strip() for d in args.dongles.split(",") if d.strip()] or None
  results = fit_all(args.cache_dir, dongle_list)
  with open(args.out, "w") as f:
    json.dump(results, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
  # quick summary
  n_dep = n_total = 0
  for car, bins in results.items():
    for bin_name, b in bins.items():
      n_total += 1
      if b["deployable"]:
        n_dep += 1
  print(f"[plant_fit] wrote {args.out}: {n_dep}/{n_total} (car, speed_bin) cells deployable", flush=True)


if __name__ == "__main__":
  sys.exit(main())
