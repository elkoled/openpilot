"""Parametric EPS plant model + train/test validation.

The model fitted here is the simplest one the data can support honestly:

    actual_curvature(v, sign) = K(v, sign) * desired_curvature

where the unit "actual" is the device-frame yaw-rate divided by vEgo (the same
quantity the sunnypilot dynamic_steering learner closes the loop on), and K is
fit per speed-bucket and per sign-of-desired separately so that left/right
asymmetry (the asymmetry the learner cannot see by design) is preserved.

What this is:
  - per-dongle K(v, sign) lookup (7 speed buckets * 2 signs = 14 numbers)
  - phase tau from the lagged cross-correlation peak
  - train/test split at the route level inside each dongle: fit on 70% of
    routes, predict on the held-out 30%, report holdout R^2
  - cross-dongle generalization: also fit a fleet K(v) and report how each
    dongle's holdout R^2 fares under the fleet model
  - bootstrap 95% CI on K via route resampling (200 reps)
  - explicit fit-failure labels for unphysical or low-confidence fits

What this is NOT:
  - a full plant transfer function with frequency response
  - a model of EPS torque dynamics (commanded MEB control is curvature, not torque;
    EPS_Lenkmoment is observable but not the control input)
  - a controller patch — that decision is downstream of the validation results
  - a guarantee of fleet generalization — the data has 9 dongles with one at 80%
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .features import N_SPEED, N_CURVATURE, CURVATURE_BUCKET_EDGES, SPEED_ANCHORS

CURVATURE_BUCKET_CENTERS = np.sqrt(CURVATURE_BUCKET_EDGES[:-1] * CURVATURE_BUCKET_EDGES[1:])
PAIR = "P1_desired_yaw"          # the model is fit on desired -> yaw-rate-derived actual
MIN_BUCKET_SAMPLES = 30          # per-bucket minimum to include in fit
MIN_ROUTES_FOR_FIT = 3           # per-dongle minimum routes to fit at all
MIN_TEST_ROUTES = 1
K_LOWER, K_UPPER = 0.2, 1.5       # physical envelope for fitted K


@dataclass
class FitResult:
  """K(v, sign) lookup + diagnostics for one fit (per-dongle or fleet)."""
  scope: str                                # 'dongle:<id>' or 'fleet'
  K_pos: np.ndarray                          # shape (N_SPEED,)  pos-desired buckets
  K_neg: np.ndarray                          # shape (N_SPEED,)
  K_pos_ci: np.ndarray                       # shape (N_SPEED, 2): (lo, hi) 95% bootstrap CI
  K_neg_ci: np.ndarray                       # shape (N_SPEED, 2)
  fit_valid_pos: np.ndarray                  # shape (N_SPEED,) bool
  fit_valid_neg: np.ndarray                  # shape (N_SPEED,) bool
  tau_s: float                                # phase lag from xcorr peak
  tau_r: float                                # peak correlation
  n_train_routes: int = 0
  n_test_routes: int = 0
  n_train_samples: int = 0
  n_test_samples: int = 0
  holdout_r2_overall: float = float("nan")
  holdout_r2_highway: float = float("nan")   # speed buckets 3..5 only
  fit_label: str = "unfitted"                # 'ok' | 'low_confidence' | 'insufficient_data'
  notes: list = field(default_factory=list)


def _pool_signed_bucket_sums(feature_dicts: Sequence[dict]) -> dict:
  """Sum per-bucket signed (count, sum_residual) across a list of per-route feature dicts."""
  out = {
    "count_pos": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
    "count_neg": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
    "sum_residual_pos": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
    "sum_residual_neg": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
    "sumsq_residual": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
    "count": np.zeros((N_SPEED, N_CURVATURE), dtype=np.float64),
  }
  for fd in feature_dicts:
    for k in out:
      key = f"{PAIR}_{k}"
      if key in fd:
        out[k] += np.asarray(fd[key], dtype=np.float64)
  return out


def _fit_K_per_speed(pooled: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """For each speed bucket s, fit K_pos[s] and K_neg[s] independently across the 12 curvature
  buckets, weighted by the per-bucket sample count.

  Model: a = K*d (no intercept). Solve K = sum(w*d*a) / sum(w*d*d).
  Treats positive-d and negative-d buckets as separate observations.
  """
  K_pos = np.full(N_SPEED, np.nan, dtype=np.float64)
  K_neg = np.full(N_SPEED, np.nan, dtype=np.float64)
  valid_pos = np.zeros(N_SPEED, dtype=bool)
  valid_neg = np.zeros(N_SPEED, dtype=bool)

  d = CURVATURE_BUCKET_CENTERS

  for s in range(N_SPEED):
    for sign, K_arr, valid_arr in ((+1, K_pos, valid_pos), (-1, K_neg, valid_neg)):
      if sign > 0:
        cnt = pooled["count_pos"][s]
        sr = pooled["sum_residual_pos"][s]
      else:
        cnt = pooled["count_neg"][s]
        sr = pooled["sum_residual_neg"][s]
      mask = cnt >= MIN_BUCKET_SAMPLES
      if int(mask.sum()) < 3:
        continue
      bucket_a = (sign * d) - np.where(cnt > 0, sr / np.where(cnt > 0, cnt, 1.0), 0.0)
      bucket_d = sign * d
      w = cnt[mask]; dd = bucket_d[mask]; aa = bucket_a[mask]
      num = float(np.sum(w * dd * aa))
      den = float(np.sum(w * dd * dd))
      if den <= 0:
        continue
      K = num / den
      if K_LOWER < K < K_UPPER:
        K_arr[s] = K
        valid_arr[s] = True
  return K_pos, K_neg, valid_pos, valid_neg


def _predict_actual(d: np.ndarray, v_idx: int, K_pos: np.ndarray, K_neg: np.ndarray,
                    valid_pos: np.ndarray, valid_neg: np.ndarray) -> np.ndarray | None:
  """Apply K(v, sign) to a vector of desireds. Returns None if K is missing at this v_idx."""
  out = np.zeros_like(d, dtype=np.float64)
  pos = d >= 0
  neg = ~pos
  if pos.any():
    if not valid_pos[v_idx]:
      return None
    out[pos] = K_pos[v_idx] * d[pos]
  if neg.any():
    if not valid_neg[v_idx]:
      return None
    out[neg] = K_neg[v_idx] * d[neg]
  return out


def _holdout_r2(test_pooled: dict, K_pos: np.ndarray, K_neg: np.ndarray,
                valid_pos: np.ndarray, valid_neg: np.ndarray,
                speed_mask: np.ndarray | None = None) -> tuple[float, int]:
  """Predict actuals on held-out bucket sums; report 1 - SS_res / SS_tot."""
  d = CURVATURE_BUCKET_CENTERS
  ss_res = 0.0; ss_tot = 0.0; n_total = 0
  speeds = range(N_SPEED) if speed_mask is None else np.flatnonzero(speed_mask)
  for s in speeds:
    for sign, cnt_arr, sr_arr in (
      (+1, test_pooled["count_pos"][s], test_pooled["sum_residual_pos"][s]),
      (-1, test_pooled["count_neg"][s], test_pooled["sum_residual_neg"][s]),
    ):
      mask = cnt_arr >= MIN_BUCKET_SAMPLES
      if not mask.any():
        continue
      d_signed = sign * d
      a_observed = d_signed - np.where(cnt_arr > 0, sr_arr / np.where(cnt_arr > 0, cnt_arr, 1.0), 0.0)
      if sign > 0:
        if not valid_pos[s]:
          continue
        a_pred = K_pos[s] * d_signed
      else:
        if not valid_neg[s]:
          continue
        a_pred = K_neg[s] * d_signed
      w = cnt_arr[mask]
      ss_res += float(np.sum(w * (a_observed[mask] - a_pred[mask]) ** 2))
      ss_tot += float(np.sum(w * (a_observed[mask] - d_signed[mask]) ** 2))
      n_total += int(w.sum())
  if ss_tot <= 0 or n_total == 0:
    return float("nan"), n_total
  return 1.0 - ss_res / ss_tot, n_total


def _bootstrap_K_ci(routes: list[dict], n_reps: int = 200, rng: np.random.Generator | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
  """Resample routes with replacement, refit K_pos/K_neg, return per-speed 95% CIs."""
  rng = rng if rng is not None else np.random.default_rng(0)
  K_pos_samples = np.full((n_reps, N_SPEED), np.nan, dtype=np.float64)
  K_neg_samples = np.full((n_reps, N_SPEED), np.nan, dtype=np.float64)
  n_routes = len(routes)
  if n_routes < 2:
    return np.full((N_SPEED, 2), np.nan), np.full((N_SPEED, 2), np.nan)
  for i in range(n_reps):
    idx = rng.integers(0, n_routes, size=n_routes)
    pooled = _pool_signed_bucket_sums([routes[j] for j in idx])
    K_p, K_n, _, _ = _fit_K_per_speed(pooled)
    K_pos_samples[i] = K_p
    K_neg_samples[i] = K_n
  K_pos_ci = np.full((N_SPEED, 2), np.nan)
  K_neg_ci = np.full((N_SPEED, 2), np.nan)
  for s in range(N_SPEED):
    pos_finite = K_pos_samples[:, s][np.isfinite(K_pos_samples[:, s])]
    neg_finite = K_neg_samples[:, s][np.isfinite(K_neg_samples[:, s])]
    if pos_finite.size >= 20:
      K_pos_ci[s] = [float(np.percentile(pos_finite, 2.5)), float(np.percentile(pos_finite, 97.5))]
    if neg_finite.size >= 20:
      K_neg_ci[s] = [float(np.percentile(neg_finite, 2.5)), float(np.percentile(neg_finite, 97.5))]
  return K_pos_ci, K_neg_ci


def _train_test_split(routes: list[dict], test_frac: float = 0.30, seed: int = 0):
  rng = np.random.default_rng(seed)
  n = len(routes)
  indices = np.arange(n)
  rng.shuffle(indices)
  n_test = max(1, int(round(n * test_frac))) if n >= 2 else 0
  test_idx = set(indices[:n_test].tolist())
  train = [r for i, r in enumerate(routes) if i not in test_idx]
  test = [r for i, r in enumerate(routes) if i in test_idx]
  return train, test


def fit_dongle(dongle_id: str, routes: list[dict], do_bootstrap: bool = True) -> FitResult:
  """Fit K(v, sign) on a 70/30 route split for one dongle."""
  result = FitResult(scope=f"dongle:{dongle_id}",
                     K_pos=np.full(N_SPEED, np.nan), K_neg=np.full(N_SPEED, np.nan),
                     K_pos_ci=np.full((N_SPEED, 2), np.nan), K_neg_ci=np.full((N_SPEED, 2), np.nan),
                     fit_valid_pos=np.zeros(N_SPEED, dtype=bool),
                     fit_valid_neg=np.zeros(N_SPEED, dtype=bool),
                     tau_s=float("nan"), tau_r=float("nan"))
  if len(routes) < MIN_ROUTES_FOR_FIT:
    result.fit_label = "insufficient_data"
    result.notes.append(f"only {len(routes)} routes, need >= {MIN_ROUTES_FOR_FIT}")
    return result

  train, test = _train_test_split(routes)
  if len(test) < MIN_TEST_ROUTES:
    result.fit_label = "insufficient_data"
    result.notes.append(f"only {len(test)} test routes after split")
    return result

  train_pooled = _pool_signed_bucket_sums(train)
  test_pooled = _pool_signed_bucket_sums(test)

  K_pos, K_neg, valid_pos, valid_neg = _fit_K_per_speed(train_pooled)
  result.K_pos = K_pos; result.K_neg = K_neg
  result.fit_valid_pos = valid_pos; result.fit_valid_neg = valid_neg

  # tau from train pool's xcorr (pooled weighted-r approach used in aggregate.pool)
  xcorr_n = np.zeros(21); xcorr_wr = np.zeros(21)
  for fd in train:
    n = np.asarray(fd.get(f"{PAIR}_xcorr_n", np.zeros(21)), dtype=np.float64)
    r = np.asarray(fd.get(f"{PAIR}_xcorr_r", np.full(21, np.nan)), dtype=np.float64)
    valid = np.isfinite(r) & (n > 0)
    xcorr_n[valid] += n[valid]; xcorr_wr[valid] += n[valid] * r[valid]
  with np.errstate(invalid="ignore", divide="ignore"):
    pooled_r = np.where(xcorr_n > 0, xcorr_wr / np.where(xcorr_n > 0, xcorr_n, 1.0), np.nan)
  if np.any(np.isfinite(pooled_r)):
    peak = int(np.nanargmax(pooled_r))
    from .features import LAG_GRID_S
    result.tau_s = float(LAG_GRID_S[peak])
    result.tau_r = float(pooled_r[peak])

  result.n_train_routes = len(train); result.n_test_routes = len(test)
  result.n_train_samples = int(train_pooled["count"].sum())
  result.n_test_samples = int(test_pooled["count"].sum())

  r2_all, _ = _holdout_r2(test_pooled, K_pos, K_neg, valid_pos, valid_neg)
  highway = np.zeros(N_SPEED, dtype=bool); highway[3:6] = True
  r2_hw, n_hw = _holdout_r2(test_pooled, K_pos, K_neg, valid_pos, valid_neg, speed_mask=highway)
  result.holdout_r2_overall = r2_all
  result.holdout_r2_highway = r2_hw

  if do_bootstrap and len(routes) >= 5:
    result.K_pos_ci, result.K_neg_ci = _bootstrap_K_ci(train, n_reps=200)

  n_valid_speeds = int(valid_pos.sum() + valid_neg.sum())
  if n_valid_speeds < 4:
    result.fit_label = "low_confidence"
    result.notes.append(f"only {n_valid_speeds}/14 speed-sign cells passed the physical envelope")
  elif not np.isfinite(r2_all) or r2_all < 0.3:
    result.fit_label = "low_confidence"
    result.notes.append(f"holdout R2 = {r2_all:.2f} is low; model does not generalize within the dongle")
  else:
    result.fit_label = "ok"
  return result


def fit_fleet_then_test_per_dongle(per_dongle_routes: dict[str, list[dict]],
                                   do_bootstrap: bool = False) -> tuple[FitResult, dict[str, float]]:
  """Pool train routes from ALL dongles into one fleet model, then for each dongle evaluate
  the fleet model on its held-out routes.

  This is the cross-dongle generalization check: if the fleet K(v) gives high holdout R^2
  on every dongle, the model is fleet-applicable; if it generalizes only on the dominant
  dongle, the model is per-dongle.
  """
  fleet_train: list[dict] = []
  per_dongle_test: dict[str, list[dict]] = {}
  per_dongle_test_pooled: dict[str, dict] = {}

  for dongle_id, routes in per_dongle_routes.items():
    if len(routes) < MIN_ROUTES_FOR_FIT:
      continue
    train, test = _train_test_split(routes)
    if not test:
      continue
    fleet_train.extend(train)
    per_dongle_test[dongle_id] = test
    per_dongle_test_pooled[dongle_id] = _pool_signed_bucket_sums(test)

  if len(fleet_train) < 10:
    fleet_fit = FitResult(scope="fleet",
                          K_pos=np.full(N_SPEED, np.nan), K_neg=np.full(N_SPEED, np.nan),
                          K_pos_ci=np.full((N_SPEED, 2), np.nan), K_neg_ci=np.full((N_SPEED, 2), np.nan),
                          fit_valid_pos=np.zeros(N_SPEED, dtype=bool),
                          fit_valid_neg=np.zeros(N_SPEED, dtype=bool),
                          tau_s=float("nan"), tau_r=float("nan"),
                          fit_label="insufficient_data")
    fleet_fit.notes.append(f"only {len(fleet_train)} fleet train routes; need >= 10")
    return fleet_fit, {}

  fleet_pooled = _pool_signed_bucket_sums(fleet_train)
  K_pos, K_neg, valid_pos, valid_neg = _fit_K_per_speed(fleet_pooled)

  fleet_fit = FitResult(
    scope="fleet",
    K_pos=K_pos, K_neg=K_neg,
    K_pos_ci=np.full((N_SPEED, 2), np.nan),
    K_neg_ci=np.full((N_SPEED, 2), np.nan),
    fit_valid_pos=valid_pos, fit_valid_neg=valid_neg,
    tau_s=float("nan"), tau_r=float("nan"),
    n_train_routes=len(fleet_train),
    n_train_samples=int(fleet_pooled["count"].sum()),
    n_test_routes=sum(len(v) for v in per_dongle_test.values()),
    n_test_samples=int(sum(p["count"].sum() for p in per_dongle_test_pooled.values())),
  )

  cross_r2: dict[str, float] = {}
  for dongle_id, pooled in per_dongle_test_pooled.items():
    r2, _ = _holdout_r2(pooled, K_pos, K_neg, valid_pos, valid_neg)
    cross_r2[dongle_id] = r2

  finite = [v for v in cross_r2.values() if np.isfinite(v)]
  if finite and min(finite) > 0.3:
    fleet_fit.fit_label = "ok"
    fleet_fit.notes.append(f"fleet model generalizes to all {len(finite)} dongles (min R2={min(finite):.2f})")
  elif finite:
    n_ok = sum(1 for r in finite if r > 0.3)
    fleet_fit.fit_label = "partial"
    fleet_fit.notes.append(f"fleet model generalizes to {n_ok}/{len(finite)} dongles only")
  else:
    fleet_fit.fit_label = "low_confidence"

  # aggregate cross-dongle R2 into fleet diagnostics
  fleet_fit.holdout_r2_overall = float(np.mean(finite)) if finite else float("nan")
  return fleet_fit, cross_r2


def write_eps_model_report(per_dongle_fits: dict[str, FitResult],
                           fleet_fit: FitResult,
                           cross_r2: dict[str, float],
                           out_path) -> None:
  from pathlib import Path
  out_path = Path(out_path)
  lines = ["# ID4 MK1 EPS plant model — train/test validation\n"]
  lines.append("## Model\n")
  lines.append("`actual_curvature = K(v_speed_bucket, sign(desired)) * desired_curvature`\n")
  lines.append("Fit independently per speed bucket and per sign of desired curvature.")
  lines.append("`K` outside `(0.2, 1.5)` is treated as a fit failure, not data.\n")
  lines.append("## Fleet model (pool train routes across all dongles, evaluate on each dongle's held-out routes)\n")
  lines.append(f"- fit label: **{fleet_fit.fit_label}**")
  lines.append(f"- train routes: {fleet_fit.n_train_routes}, train samples: {fleet_fit.n_train_samples}")
  lines.append(f"- test routes (across dongles): {fleet_fit.n_test_routes}, test samples: {fleet_fit.n_test_samples}")
  for n in fleet_fit.notes:
    lines.append(f"- note: {n}")
  lines.append("\n### Fleet K(v) (pos / neg)\n")
  lines.append("| v (kph) | K_pos | K_neg | valid |")
  lines.append("|---------|-------|-------|-------|")
  for s, v in enumerate(SPEED_ANCHORS):
    kp = fleet_fit.K_pos[s]; kn = fleet_fit.K_neg[s]
    valid = "yes" if (fleet_fit.fit_valid_pos[s] and fleet_fit.fit_valid_neg[s]) else "no"
    lines.append(f"| {int(v*3.6):3d} | {kp:.3f} | {kn:.3f} | {valid} |")
  lines.append("\n### Cross-dongle generalization (fleet model holdout R^2 per dongle)\n")
  lines.append("| dongle | holdout R^2 |")
  lines.append("|--------|-------------|")
  for d, r in sorted(cross_r2.items(), key=lambda kv: -kv[1] if np.isfinite(kv[1]) else 1):
    rs = f"{r:.3f}" if np.isfinite(r) else "nan"
    lines.append(f"| {d} | {rs} |")
  lines.append("\n## Per-dongle K(v) fits\n")
  for dongle_id, fit in sorted(per_dongle_fits.items(), key=lambda kv: kv[0]):
    lines.append(f"### {dongle_id} ({fit.fit_label})\n")
    lines.append(f"- routes: {fit.n_train_routes} train / {fit.n_test_routes} test")
    lines.append(f"- holdout R^2 overall: {fit.holdout_r2_overall:.3f}")
    lines.append(f"- holdout R^2 highway (80-120 kph): {fit.holdout_r2_highway:.3f}")
    if np.isfinite(fit.tau_s):
      lines.append(f"- phase tau (xcorr peak): {fit.tau_s*1000:+.0f} ms (r={fit.tau_r:.3f})")
    for n in fit.notes:
      lines.append(f"- note: {n}")
    lines.append("\n| v (kph) | K_pos | 95% CI | K_neg | 95% CI |")
    lines.append("|---------|-------|--------|-------|--------|")
    for s, v in enumerate(SPEED_ANCHORS):
      kp = fit.K_pos[s]; kn = fit.K_neg[s]
      cip = fit.K_pos_ci[s]; cin = fit.K_neg_ci[s]
      kp_s = f"{kp:.3f}" if fit.fit_valid_pos[s] else "—"
      kn_s = f"{kn:.3f}" if fit.fit_valid_neg[s] else "—"
      cip_s = f"[{cip[0]:.2f}, {cip[1]:.2f}]" if np.isfinite(cip[0]) else "—"
      cin_s = f"[{cin[0]:.2f}, {cin[1]:.2f}]" if np.isfinite(cin[0]) else "—"
      lines.append(f"| {int(v*3.6):3d} | {kp_s} | {cip_s} | {kn_s} | {cin_s} |")
    lines.append("")
  out_path.write_text("\n".join(lines) + "\n")
