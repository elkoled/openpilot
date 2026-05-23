"""Alternative-hypothesis scoring for per-dongle pooled features.

Fits each candidate hypothesis to the pooled bucket statistics, returns a fit
result with AIC, BIC, and a fit-failure flag. The point is to avoid the prior-
session failure mode where a single "K > 1" or "tau is null" was treated as
data; here unphysical fits are explicitly labelled as failures, not data.

Input: a `BucketSummary` (see aggregate.py) summarizing pooled per-bucket
count, sum_residual, sumsq_residual, sign-stratified means, plus the lagged
xcorr correlations. We do not need raw samples — bucket-pooled statistics are
sufficient for the fits at the level of resolution this analysis cares about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import CURVATURE_BUCKET_EDGES, LAG_GRID_S, SPEED_ANCHORS


CURVATURE_BUCKET_CENTERS = np.sqrt(CURVATURE_BUCKET_EDGES[:-1] * CURVATURE_BUCKET_EDGES[1:])


@dataclass
class HypothesisFit:
  name: str
  params: dict
  rss: float       # residual sum of squares (lower is better)
  n: int           # samples used
  k: int           # free parameters
  failed: bool = False
  reason: str = ""

  @property
  def aic(self) -> float:
    if self.failed or self.n <= self.k + 1:
      return float("inf")
    sigma2 = max(self.rss / self.n, 1e-30)
    return self.n * np.log(sigma2) + 2.0 * self.k

  @property
  def bic(self) -> float:
    if self.failed or self.n <= self.k + 1:
      return float("inf")
    sigma2 = max(self.rss / self.n, 1e-30)
    return self.n * np.log(sigma2) + self.k * np.log(self.n)


def _bucket_points(count: np.ndarray, sum_residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Treat each bucket as one synthetic (desired, actual) point.
  Returns desireds[7,12], actuals[7,12], counts[7,12], with sign-stratification not applied here
  (use _signed_bucket_points for that).
  """
  desireds = CURVATURE_BUCKET_CENTERS[None, :].repeat(len(SPEED_ANCHORS), axis=0)
  with np.errstate(divide="ignore", invalid="ignore"):
    mean_residual = np.where(count > 0, sum_residual / np.where(count > 0, count, 1.0), 0.0)
  actuals = desireds - mean_residual
  return desireds, actuals, count


def _signed_bucket_points(count_pos, count_neg, sum_residual_pos, sum_residual_neg):
  """Return separately the +desired and -desired bucket means."""
  d = CURVATURE_BUCKET_CENTERS[None, :].repeat(len(SPEED_ANCHORS), axis=0)
  with np.errstate(divide="ignore", invalid="ignore"):
    pos_mean_res = np.where(count_pos > 0, sum_residual_pos / np.where(count_pos > 0, count_pos, 1.0), 0.0)
    neg_mean_res = np.where(count_neg > 0, sum_residual_neg / np.where(count_neg > 0, count_neg, 1.0), 0.0)
  pos_d = d
  neg_d = -d
  pos_a = pos_d - pos_mean_res
  neg_a = neg_d - neg_mean_res
  return pos_d, pos_a, count_pos, neg_d, neg_a, count_neg


def _flatten_valid(d: np.ndarray, a: np.ndarray, w: np.ndarray, min_per_bucket: int = 8):
  mask = (w >= min_per_bucket)
  return d[mask].astype(np.float64), a[mask].astype(np.float64), w[mask].astype(np.float64)


def fit_null(count, sum_residual, sumsq_residual) -> HypothesisFit:
  d, a, w = _bucket_points(count, sum_residual)
  d_f, a_f, w_f = _flatten_valid(d, a, w)
  if d_f.size == 0:
    return HypothesisFit("H_null", {}, 0.0, 0, 0, failed=True, reason="no buckets with enough samples")
  # H_null: actual = desired (K=1 perfectly); RSS = sum_bucket weight * (a - d)^2.
  # Use the actual per-sample sumsq (more accurate than bucket-mean approximation).
  total_n = int(count.sum())
  rss = float(sumsq_residual.sum())
  return HypothesisFit("H_null", {}, rss, total_n, 0)


def fit_gain(count_pos, count_neg, sum_residual_pos, sum_residual_neg) -> HypothesisFit:
  """Combine sign-stratified buckets into signed (d, a) synthetic points so that
  positive and negative residuals don't cancel in the unsigned bucket sums.
  Treat positive-d and negative-d buckets as separate observations at d=+c and d=-c."""
  pos_d, pos_a, pos_w, neg_d, neg_a, neg_w = _signed_bucket_points(
    count_pos, count_neg, sum_residual_pos, sum_residual_neg)
  pos_d_f, pos_a_f, pos_w_f = _flatten_valid(pos_d, pos_a, pos_w)
  neg_d_f, neg_a_f, neg_w_f = _flatten_valid(neg_d, neg_a, neg_w)
  d_f = np.concatenate([pos_d_f, neg_d_f])
  a_f = np.concatenate([pos_a_f, neg_a_f])
  w_f = np.concatenate([pos_w_f, neg_w_f])
  if d_f.size < 3:
    return HypothesisFit("H_gain", {}, 0.0, 0, 1, failed=True, reason="<3 valid signed buckets")
  num = float(np.sum(w_f * d_f * a_f))
  den = float(np.sum(w_f * d_f * d_f))
  if den <= 0:
    return HypothesisFit("H_gain", {}, 0.0, 0, 1, failed=True, reason="degenerate")
  K = num / den
  if not (0.2 < K < 1.5):
    return HypothesisFit("H_gain", {"K": K}, 0.0, int(w_f.sum()), 1, failed=True, reason=f"unphysical K={K:.3f}")
  resid = a_f - K * d_f
  rss = float(np.sum(w_f * resid * resid))
  return HypothesisFit("H_gain", {"K": K}, rss, int(w_f.sum()), 1)


def fit_lag(xcorr_r: np.ndarray, gated_n: int) -> HypothesisFit:
  if not np.any(np.isfinite(xcorr_r)):
    return HypothesisFit("H_lag", {}, 0.0, 0, 1, failed=True, reason="xcorr all nan")
  peak_idx = int(np.nanargmax(xcorr_r))
  r_peak = float(xcorr_r[peak_idx])
  tau = float(LAG_GRID_S[peak_idx])
  if r_peak < 0.5:
    return HypothesisFit("H_lag", {"tau_s": tau, "r": r_peak}, 0.0, gated_n, 1,
                         failed=True, reason=f"weak xcorr r={r_peak:.3f}")
  # Pseudo-RSS: fraction of variance NOT explained at the best lag.
  rss_proxy = gated_n * (1.0 - r_peak * r_peak)
  return HypothesisFit("H_lag", {"tau_s": tau, "r": r_peak}, rss_proxy, gated_n, 1)


def fit_asymmetry(count_pos, count_neg, sum_residual_pos, sum_residual_neg) -> HypothesisFit:
  pos_d, pos_a, pos_w, neg_d, neg_a, neg_w = _signed_bucket_points(
    count_pos, count_neg, sum_residual_pos, sum_residual_neg)
  pos_d_f, pos_a_f, pos_w_f = _flatten_valid(pos_d, pos_a, pos_w)
  neg_d_f, neg_a_f, neg_w_f = _flatten_valid(neg_d, neg_a, neg_w)
  if pos_d_f.size < 3 or neg_d_f.size < 3:
    return HypothesisFit("H_asymmetry", {}, 0.0, 0, 2, failed=True,
                         reason="<3 valid buckets in one sign")
  def _fit_K(d, a, w):
    num = float(np.sum(w * d * a)); den = float(np.sum(w * d * d))
    return num / den if den > 0 else float("nan")
  K_pos = _fit_K(pos_d_f, pos_a_f, pos_w_f)
  K_neg = _fit_K(neg_d_f, neg_a_f, neg_w_f)
  if not (0.2 < K_pos < 1.5) or not (0.2 < K_neg < 1.5):
    return HypothesisFit("H_asymmetry", {"K_pos": K_pos, "K_neg": K_neg}, 0.0,
                         int(pos_w_f.sum() + neg_w_f.sum()), 2, failed=True,
                         reason=f"unphysical K_pos={K_pos:.2f} K_neg={K_neg:.2f}")
  # Reject asymmetry hypothesis when the asymmetry is below an empirical detection threshold.
  # Without this, the extra free param can overfit shot noise and beat H_gain in AIC even
  # when the underlying truth is symmetric.
  delta = abs(K_pos - K_neg)
  if delta < 0.05:
    return HypothesisFit("H_asymmetry", {"K_pos": K_pos, "K_neg": K_neg, "delta": delta}, 0.0,
                         int(pos_w_f.sum() + neg_w_f.sum()), 2, failed=True,
                         reason=f"asymmetry below detection threshold (delta={delta:.3f})")
  rss_pos = float(np.sum(pos_w_f * (pos_a_f - K_pos * pos_d_f) ** 2))
  rss_neg = float(np.sum(neg_w_f * (neg_a_f - K_neg * neg_d_f) ** 2))
  n = int(pos_w_f.sum() + neg_w_f.sum())
  return HypothesisFit("H_asymmetry", {"K_pos": K_pos, "K_neg": K_neg, "delta": delta},
                       rss_pos + rss_neg, n, 2)


def fit_deadband(count, sum_residual_pos, sum_residual_neg, count_pos, count_neg,
                 deadband_count, deadband_sum_residual) -> HypothesisFit:
  """Deadband signature: |residual|/|desired| ratio is much higher in the near-zero stratum
  than in the larger-|desired| strata. Comparison uses the abs-mean residual *per unit
  desired*, so a true deadband (constant offset) stands out clearly while a pure gain
  mismatch (proportional residual) does not.
  """
  if deadband_count[0] < 30:
    return HypothesisFit("H_deadband", {}, 0.0, 0, 2, failed=True, reason="too few near-zero samples")
  # mean |residual| in the near-zero stratum
  near_zero_abs_mean = float(abs(deadband_sum_residual[0]) / deadband_count[0])
  # mean |residual| outside the near-zero stratum, using sign-stratified buckets
  # (so pos/neg residuals don't cancel). Excludes the smallest-|d| curvature buckets.
  outer_curv_mask = CURVATURE_BUCKET_CENTERS >= 1.0e-5
  total_outer = float(count_pos[:, outer_curv_mask].sum() + count_neg[:, outer_curv_mask].sum())
  total_outer_signed = float(
    abs(sum_residual_pos[:, outer_curv_mask]).sum() + abs(sum_residual_neg[:, outer_curv_mask]).sum()
  )
  outer_abs_mean = total_outer_signed / total_outer if total_outer > 0 else 0.0
  if near_zero_abs_mean < 5.0e-5 or near_zero_abs_mean < 3.0 * (outer_abs_mean + 1e-12):
    return HypothesisFit("H_deadband", {"near_zero": near_zero_abs_mean, "outer": outer_abs_mean}, 0.0,
                         int(deadband_count[0]), 2, failed=True, reason="no deadband signature")
  rss_proxy = float((deadband_sum_residual[0] ** 2) / max(deadband_count[0], 1.0))
  return HypothesisFit("H_deadband", {"d_est": near_zero_abs_mean, "outer": outer_abs_mean},
                       rss_proxy, int(deadband_count[0]), 2)


def rank_hypotheses(fits: list[HypothesisFit]) -> tuple[str, dict]:
  """Return (winner_name, {hypothesis_name: aic}). Winner = lowest AIC. If ties (|Δaic| < 2),
  return 'mixed'.
  """
  scored = [(f.name, f.aic, f.failed) for f in fits]
  active = [(n, a) for n, a, fail in scored if not fail and np.isfinite(a)]
  aic_map = {n: a for n, a, _ in scored}
  if not active:
    return "all_failed", aic_map
  active.sort(key=lambda x: x[1])
  if len(active) == 1:
    return active[0][0], aic_map
  best, second = active[0], active[1]
  if (second[1] - best[1]) < 2.0:
    return "mixed", aic_map
  return best[0], aic_map


def score_dongle(features: dict) -> dict:
  """Run all hypotheses on a per-dongle pooled feature dict (same shape as a per-route dict,
  since pooled buckets have identical schema). Returns the winning hypothesis + per-hypothesis
  details, plus an explicit fit-failure list.
  """
  label = "P1_desired_yaw"
  fits = [
    fit_null(features[f"{label}_count"], features[f"{label}_sum_residual"], features[f"{label}_sumsq_residual"]),
    fit_gain(features[f"{label}_count_pos"], features[f"{label}_count_neg"],
             features[f"{label}_sum_residual_pos"], features[f"{label}_sum_residual_neg"]),
    fit_lag(features[f"{label}_xcorr_r"], int(features[f"{label}_count"].sum())),
    fit_asymmetry(features[f"{label}_count_pos"], features[f"{label}_count_neg"],
                  features[f"{label}_sum_residual_pos"], features[f"{label}_sum_residual_neg"]),
    fit_deadband(features[f"{label}_count"],
                 features[f"{label}_sum_residual_pos"], features[f"{label}_sum_residual_neg"],
                 features[f"{label}_count_pos"], features[f"{label}_count_neg"],
                 features[f"{label}_deadband_count"], features[f"{label}_deadband_sum_residual"]),
  ]
  winner, aic_map = rank_hypotheses(fits)
  return {
    "winner": winner,
    "fits": {f.name: {"params": f.params, "aic": f.aic, "bic": f.bic, "failed": f.failed, "reason": f.reason, "n": f.n}
             for f in fits},
    "aic_map": aic_map,
  }
