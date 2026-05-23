"""Per-dongle and fleet aggregation.

Pools per-route feature dicts (all keyed by the schemas defined in features.py)
into per-dongle and per-fleet feature dicts of the same shape. Then runs the
hypothesis scorer on each dongle.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .extract import unpack_features
from .hypotheses import score_dongle, CURVATURE_BUCKET_CENTERS
from .features import LAG_GRID_S, SPEED_ANCHORS, N_SPEED, N_CURVATURE


_BUCKET_SUM_KEYS = (
  "count", "sum_residual", "sum_abs_residual", "sumsq_residual",
  "count_pos", "count_neg", "sum_residual_pos", "sum_residual_neg",
  "saturation_count",
  "deadband_count", "deadband_sum_residual", "deadband_sumsq_residual",
)
_XCORR_SUM_KEYS = ("n", "sxy")     # we re-derive r from these


PAIR_LABELS = ("P1_desired_yaw", "P2_desired_qfk", "P3_apply_yaw", "P4_apply_qfk")


def _zero_features() -> dict:
  out: dict = {}
  shape = (N_SPEED, N_CURVATURE)
  for label in PAIR_LABELS:
    for k in _BUCKET_SUM_KEYS:
      if k.startswith("deadband"):
        out[f"{label}_{k}"] = np.zeros(3, dtype=np.float64)
      else:
        out[f"{label}_{k}"] = np.zeros(shape, dtype=np.float64)
    out[f"{label}_xcorr_n"] = np.zeros(len(LAG_GRID_S), dtype=np.float64)
    out[f"{label}_xcorr_sxy"] = np.zeros(len(LAG_GRID_S), dtype=np.float64)
  return out


def pool(feature_dicts: list[dict]) -> dict:
  """Sum same-key arrays across a list of per-route feature dicts."""
  if not feature_dicts:
    return _zero_features()
  out = _zero_features()
  for fd in feature_dicts:
    for k, v in out.items():
      if k in fd:
        out[k] = v + fd[k]
  # rebuild xcorr_r from pooled n and sxy if we have raw sx/sy/sxx/syy (we don't here;
  # the per-route x/y stats live in the original XcorrAccumulator and aren't summable
  # across routes safely without storing more state. For the pooled correlation we use
  # the per-route weighted average of correlation, weighted by xcorr_n).
  for label in PAIR_LABELS:
    weighted_r = np.zeros(len(LAG_GRID_S), dtype=np.float64)
    total_w = np.zeros(len(LAG_GRID_S), dtype=np.float64)
    for fd in feature_dicts:
      key_r = f"{label}_xcorr_r"; key_n = f"{label}_xcorr_n"
      if key_r in fd and key_n in fd:
        r = np.asarray(fd[key_r], dtype=np.float64)
        n = np.asarray(fd[key_n], dtype=np.float64)
        valid = np.isfinite(r) & (n > 0)
        weighted_r[valid] += (n[valid] * r[valid])
        total_w[valid] += n[valid]
    with np.errstate(invalid="ignore", divide="ignore"):
      out[f"{label}_xcorr_r"] = np.where(total_w > 0, weighted_r / np.where(total_w > 0, total_w, 1.0), np.nan)
  return out


@dataclass
class DongleSummary:
  dongle_id: str
  n_routes: int
  total_seconds: float
  engaged_seconds: float
  gated_samples: int
  scalar_tracking_score: float       # RMS bucket residual in highway-gentle window, lower=better
  fingerprint: str
  vin: str
  hypothesis: dict                   # output of score_dongle()
  pooled_features: dict


def _scalar_tracking_score(pooled: dict) -> float:
  """RMS residual in lat-accel<1 m/s^2 x speed in {3,4,5} (~80..120 kph) window of P1."""
  label = "P1_desired_yaw"
  c = pooled[f"{label}_count"]
  sq = pooled[f"{label}_sumsq_residual"]
  n = float(c[3:6].sum())
  if n <= 0:
    return float("nan")
  s = float(sq[3:6].sum())
  return float(np.sqrt(max(s / n, 0.0)))


def aggregate_per_dongle(per_route_rows: list[dict]) -> list[DongleSummary]:
  """Each per_route_rows entry has keys: dongle_id, route_id, branch, status,
  car_fingerprint, car_vin, engaged_seconds, total_seconds, n_samples_gated, features (dict).
  Only status='ok' rows contribute to pooling; others contribute to counts only.
  """
  by_dongle: dict[str, list[dict]] = defaultdict(list)
  for row in per_route_rows:
    if row.get("status") == "ok":
      by_dongle[row["dongle_id"]].append(row)

  summaries: list[DongleSummary] = []
  for dongle_id, routes in by_dongle.items():
    pooled = pool([r["features"] for r in routes])
    total_s = sum(r.get("total_seconds", 0.0) for r in routes)
    eng_s = sum(r.get("engaged_seconds", 0.0) for r in routes)
    gated = int(sum(r.get("n_samples_gated", 0) for r in routes))
    fp = routes[0].get("car_fingerprint", "")
    vin = routes[0].get("car_vin", "")
    hyp = score_dongle(pooled)
    score = _scalar_tracking_score(pooled)
    summaries.append(DongleSummary(
      dongle_id=dongle_id,
      n_routes=len(routes),
      total_seconds=total_s,
      engaged_seconds=eng_s,
      gated_samples=gated,
      scalar_tracking_score=score,
      fingerprint=fp,
      vin=vin,
      hypothesis=hyp,
      pooled_features=pooled,
    ))

  summaries.sort(key=lambda s: (s.scalar_tracking_score if np.isfinite(s.scalar_tracking_score) else 1e9))
  return summaries


def fleet_decision(dongle_summaries: list[DongleSummary], min_dongles: int = 10) -> dict:
  """Top-level decision summary. Returns a dict suitable for serializing to decision.md."""
  finite = [d for d in dongle_summaries if np.isfinite(d.scalar_tracking_score) and d.gated_samples >= 200]
  decision = {
    "n_dongles_total": len(dongle_summaries),
    "n_dongles_qualified": len(finite),
    "min_dongles_for_fleet_conclusion": min_dongles,
    "fleet_conclusion_valid": len(finite) >= min_dongles,
    "warnings": [],
    "per_hypothesis_count": {},
    "recommendation": "insufficient_data",
  }
  if len(finite) < min_dongles:
    decision["warnings"].append(
      f"only {len(finite)} dongles passed the gated-samples threshold (need >= {min_dongles}); "
      "treat all fleet-level statements as preliminary."
    )

  winner_counts: dict[str, int] = defaultdict(int)
  for d in finite:
    winner_counts[d.hypothesis["winner"]] += 1
  decision["per_hypothesis_count"] = dict(winner_counts)

  scores = np.array([d.scalar_tracking_score for d in finite], dtype=np.float64)
  if scores.size > 0:
    decision["score_percentiles"] = {p: float(np.percentile(scores, p)) for p in (5, 25, 50, 75, 95)}
    iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    decision["score_iqr"] = iqr
    decision["score_iqr_over_median_ratio"] = float(iqr / max(np.percentile(scores, 50), 1e-12))

  if decision["fleet_conclusion_valid"]:
    iqr_ratio = decision.get("score_iqr_over_median_ratio", 0.0)
    if iqr_ratio < 0.20:
      decision["recommendation"] = "no_layer_needed"
      decision["rationale"] = (
        f"Cross-dongle scalar tracking score IQR/median = {iqr_ratio:.2f} < 0.20. "
        "Tracking quality is tight across the fleet; no per-car layer is justified."
      )
    elif winner_counts.get("H_gain", 0) >= 0.6 * len(finite):
      decision["recommendation"] = "per_dongle_gain_scalar"
      decision["rationale"] = (
        f"{winner_counts['H_gain']}/{len(finite)} dongles' residuals are best explained by a "
        f"steady-state gain K. A per-dongle scalar gain is the smallest defensible intervention."
      )
    elif winner_counts.get("H_lag", 0) >= 0.4 * len(finite):
      decision["recommendation"] = "fix_delay_estimator"
      decision["rationale"] = (
        f"{winner_counts['H_lag']}/{len(finite)} dongles' residuals are best explained by phase lag. "
        "Investigate the lateral-delay estimator before adding controller-side correction."
      )
    elif winner_counts.get("H_asymmetry", 0) >= 0.3 * len(finite):
      decision["recommendation"] = "negative_result_asymmetry"
      decision["rationale"] = (
        f"{winner_counts['H_asymmetry']}/{len(finite)} dongles show left/right asymmetry, which the "
        "existing dynamic_steering learner cannot capture by design (signs are folded). No thin "
        "controller-side layer can fix this without rethinking the learner."
      )
    elif winner_counts.get("H_null", 0) >= 0.6 * len(finite):
      decision["recommendation"] = "no_layer_needed"
      decision["rationale"] = "Most dongles show no significant residual; bad cars are outliers, not the rule."
    else:
      decision["recommendation"] = "mixed_no_clean_layer"
      decision["rationale"] = (
        "No single hypothesis explains the fleet residual. A thin per-car layer is unlikely to "
        "close the gap without per-dongle tuning that won't scale. Consider per-VIN calibration "
        "reset or further investigation."
      )
  return decision


def load_per_route_rows(parquet_path: str) -> list[dict]:
  import pyarrow.parquet as pq
  table = pq.read_table(parquet_path)
  rows: list[dict] = []
  cols = {c: table.column(c).to_pylist() for c in table.column_names}
  n = len(cols.get("route_key", []))
  for i in range(n):
    row = {c: cols[c][i] for c in cols}
    blob = row.get("features_blob")
    if blob is not None and len(blob) > 0:
      try:
        row["features"] = unpack_features(blob)
      except Exception:
        row["features"] = {}
    else:
      row["features"] = {}
    rows.append(row)
  return rows
