"""
Aggregation grids for ID4_MK1 residual analysis.

We use two grids:

  - LEARNER grid: identical to the openpilot10 dynamic_steering learner
    (7 speeds 20..140 km/h, 12 log-spaced curvature buckets 1e-6..4.096e-3
    rad/m). This is where the production learner can converge to a bias.

  - EXTENDED grid: same speed anchors plus 150 and 160 km/h, plus 4 more
    log-spaced curvature buckets out to 6.554e-2 rad/m. This lets us measure
    whether residual mass concentrates inside the learner-supported region
    or outside it — one of the explicit open questions.
"""
from __future__ import annotations

import numpy as np

# Speed anchors in m/s
LEARNER_SPEED_ANCHORS = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=np.float32) / 3.6
EXTENDED_SPEED_ANCHORS = np.array(
  [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 150.0, 160.0], dtype=np.float32) / 3.6

LEARNER_CURVATURE_EDGES = np.array([
  1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5,
  1.28e-4, 2.56e-4, 5.12e-4, 1.024e-3, 2.048e-3, 4.096e-3,
], dtype=np.float32)

# Continues the log-2 spacing past the learner's last edge.
EXTENDED_CURVATURE_EDGES = np.concatenate([
  LEARNER_CURVATURE_EDGES,
  np.array([8.192e-3, 1.6384e-2, 3.2768e-2, 6.5536e-2], dtype=np.float32),
])

LEARNER_SHAPE = (len(LEARNER_SPEED_ANCHORS), len(LEARNER_CURVATURE_EDGES) - 1)
EXTENDED_SHAPE = (len(EXTENDED_SPEED_ANCHORS), len(EXTENDED_CURVATURE_EDGES) - 1)


def speed_bucket(v_ego: float, anchors: np.ndarray) -> int | None:
  """Nearest-anchor speed bucket. Returns None if v_ego < anchor[0] / 2."""
  v = float(v_ego)
  if v < float(anchors[0]) * 0.5:
    return None
  return int(np.argmin(np.abs(anchors - v)))


def curvature_bucket(curvature: float, edges: np.ndarray) -> int | None:
  abs_c = abs(float(curvature))
  if abs_c < edges[0] or abs_c > edges[-1]:
    return None
  idx = int(np.searchsorted(edges, abs_c, side="right") - 1)
  return min(max(idx, 0), len(edges) - 2)


class BucketAccumulator:
  """Streaming bucket aggregator. Holds only per-bucket running sums, never the
  raw timeline. Memory ~ S * C * 8 floats ≈ a few KB."""

  def __init__(self, shape: tuple[int, int]):
    S, C = shape
    self.shape = shape
    self.count = np.zeros((S, C), dtype=np.float64)
    self.sum_err = np.zeros((S, C), dtype=np.float64)         # signed
    self.sum_err_sq = np.zeros((S, C), dtype=np.float64)
    self.sum_abs_err = np.zeros((S, C), dtype=np.float64)
    self.sum_err_left = np.zeros((S, C), dtype=np.float64)    # desired > 0
    self.cnt_left = np.zeros((S, C), dtype=np.float64)
    self.sum_err_right = np.zeros((S, C), dtype=np.float64)   # desired < 0
    self.cnt_right = np.zeros((S, C), dtype=np.float64)
    self.sum_roll = np.zeros((S, C), dtype=np.float64)

  def add(self, s_idx: int, c_idx: int, signed_err: float, desired_sign: int, roll: float):
    self.count[s_idx, c_idx] += 1.0
    self.sum_err[s_idx, c_idx] += signed_err
    self.sum_err_sq[s_idx, c_idx] += signed_err * signed_err
    self.sum_abs_err[s_idx, c_idx] += abs(signed_err)
    self.sum_roll[s_idx, c_idx] += roll
    if desired_sign > 0:
      self.sum_err_left[s_idx, c_idx] += signed_err
      self.cnt_left[s_idx, c_idx] += 1.0
    elif desired_sign < 0:
      self.sum_err_right[s_idx, c_idx] += signed_err
      self.cnt_right[s_idx, c_idx] += 1.0

  def to_dict(self) -> dict:
    return {
      "count": self.count.tolist(),
      "sum_err": self.sum_err.tolist(),
      "sum_err_sq": self.sum_err_sq.tolist(),
      "sum_abs_err": self.sum_abs_err.tolist(),
      "sum_err_left": self.sum_err_left.tolist(),
      "cnt_left": self.cnt_left.tolist(),
      "sum_err_right": self.sum_err_right.tolist(),
      "cnt_right": self.cnt_right.tolist(),
      "sum_roll": self.sum_roll.tolist(),
      "shape": list(self.shape),
    }

  @classmethod
  def from_dict(cls, d: dict) -> "BucketAccumulator":
    a = cls(tuple(d["shape"]))
    for k in ("count", "sum_err", "sum_err_sq", "sum_abs_err",
              "sum_err_left", "cnt_left", "sum_err_right", "cnt_right", "sum_roll"):
      setattr(a, k, np.array(d[k], dtype=np.float64))
    return a

  def __iadd__(self, other: "BucketAccumulator") -> "BucketAccumulator":
    assert self.shape == other.shape
    for k in ("count", "sum_err", "sum_err_sq", "sum_abs_err",
              "sum_err_left", "cnt_left", "sum_err_right", "cnt_right", "sum_roll"):
      setattr(self, k, getattr(self, k) + getattr(other, k))
    return self
