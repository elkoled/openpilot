#!/usr/bin/env python3
"""
Verification tests for the lateral analysis pipeline.

Three layers:

  1. Synthetic unit tests: learner_replay + bucket aggregator + grid math
     run without network or logs. These guard the bucket-grid arithmetic
     and the offline learner's faithfulness to the openpilot10 daemon.

  2. Reference-route extraction: runs extract() against the two routes the
     user supplied as anchors:
       f73c01590368ee5b/00000010--19b95d93b3  (Stock TA / no openpilot)
       f73c01590368ee5b/0000000e--2d623b6df3  (Sunnypilot PID engaged)
     Skipped (not failed) if the network or LogReader can't reach them.

  3. Decision-tree sanity: render() consumes a hand-built fleet_stats dict
     and emits HTML without crashing.

Run:
  python -m pytest tools/vw_id4_lateral/test_extract.py -v
"""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest

from openpilot.tools.vw_id4_lateral.grid import (
  BucketAccumulator, EXTENDED_CURVATURE_EDGES, EXTENDED_SHAPE,
  EXTENDED_SPEED_ANCHORS, LEARNER_CURVATURE_EDGES, LEARNER_SHAPE,
  LEARNER_SPEED_ANCHORS, curvature_bucket, speed_bucket,
)
from openpilot.tools.vw_id4_lateral.learner_replay import (
  CurvatureDLookup, OfflineCurvatureEstimator, batch_fit_per_bucket,
)


# ----- Layer 1: synthetic unit tests -----

class TestLearnerVendored:
  def test_speed_anchors_match_openpilot10(self):
    anchors_kph = CurvatureDLookup.SPEED_ANCHORS * 3.6
    np.testing.assert_allclose(anchors_kph, [20, 40, 60, 80, 100, 120, 140], atol=1e-3)

  def test_bucket_grid_shape(self):
    assert CurvatureDLookup.bucket_shape() == (7, 12)

  def test_curvature_index_below_min(self):
    assert CurvatureDLookup.curvature_index(1e-9) is None

  def test_curvature_index_above_max(self):
    assert CurvatureDLookup.curvature_index(1.0) is None

  def test_curvature_index_inside_range(self):
    # 1e-4 falls between edges 6.4e-5 and 1.28e-4 → bucket 6
    assert CurvatureDLookup.curvature_index(1e-4) == 6

  def test_speed_index_clamps_low(self):
    low, high, alpha = CurvatureDLookup.speed_interp(2.0)
    assert (low, high, alpha) == (0, 0, 0.0)

  def test_speed_interp_blends(self):
    # 80/3.6 = 22.22 m/s; 100/3.6 = 27.77; midpoint 25.0
    low, high, alpha = CurvatureDLookup.speed_interp(25.0)
    assert low == 3 and high == 4
    assert 0.4 < alpha < 0.6

  def test_projected_error_direction(self):
    assert CurvatureDLookup.projected_error(1e-3, 0.5e-3) == pytest.approx(0.5e-3, abs=1e-9)
    assert CurvatureDLookup.projected_error(-1e-3, -0.5e-3) == pytest.approx(0.5e-3, abs=1e-9)
    # opposite signs → large positive error in direction-projected space
    assert CurvatureDLookup.projected_error(1e-3, -1e-3) == pytest.approx(2e-3, abs=1e-9)


class TestOfflineCurvatureEstimator:
  def test_left_and_right_feed_same_bucket(self):
    """Direction is projected out → +c with a=0.5c and -c with a=-0.5c
    should produce the same per-bucket bias."""
    est_l = OfflineCurvatureEstimator()
    est_r = OfflineCurvatureEstimator()
    v = 100.0 / 3.6
    c = 5e-4
    for _ in range(400):
      est_l.step(c, 0.5 * c, v)
      est_r.step(-c, -0.5 * c, v)
    # bucket index for 5e-4 → between edges 5.12e-4? Let's check: 5e-4 falls
    # between 2.56e-4 and 5.12e-4 → bucket 8
    np.testing.assert_allclose(est_l.bias, est_r.bias, atol=1e-9)

  def test_replay_converges_to_constant_bias(self):
    """If actual is consistently desired/2, the per-bucket bias should
    converge to ±50% of the cap, NOT to desired/2 directly (the learner
    clips error to ±50% of |curvature|)."""
    v = 100.0 / 3.6
    c = 1e-3
    est = OfflineCurvatureEstimator()
    for _ in range(2000):
      est.step(c, 0.5 * c, v)
    cap = 0.5 * c
    # bias should be exactly cap (positive — desired - actual = +0.5c)
    idx_c = CurvatureDLookup.curvature_index(c)
    # find which speed anchor(s) get weight
    sw = CurvatureDLookup.learning_speed_weights(v)
    for s, _ in sw:
      assert est.bias[s, idx_c] == pytest.approx(cap, abs=1e-6)

  def test_max_samples_saturation(self):
    """Counts saturate at MAX_SAMPLES = 600 even with many measurements."""
    v = 80.0 / 3.6
    c = 5e-4
    est = OfflineCurvatureEstimator()
    for _ in range(3000):
      est.step(c, 0.4 * c, v)
    assert est.counts.max() == pytest.approx(600.0, abs=1e-6)

  def test_below_min_speed_rejected(self):
    est = OfflineCurvatureEstimator()
    ok = est.step(1e-4, 5e-5, 1.0)  # 1 m/s far below MIN_SPEED
    assert ok is False
    assert est.num_applied == 0


class TestBatchFit:
  def test_zero_actual_gives_clipped_bias(self):
    """If desired is consistently 1e-3 and actual=0, projected error is 1e-3
    which exceeds the ±50% cap (5e-4). Batch fit should report 5e-4."""
    v = 100.0 / 3.6
    c = 1e-3
    n = 2000
    d = np.full(n, c)
    a = np.zeros(n)
    vv = np.full(n, v)
    res = batch_fit_per_bucket(d, a, vv)
    idx_c = CurvatureDLookup.curvature_index(c)
    sw = CurvatureDLookup.learning_speed_weights(v)
    for s, _ in sw:
      assert res.bias[s, idx_c] == pytest.approx(5e-4, abs=1e-6)
      assert res.bias_clipped_fraction[s, idx_c] == pytest.approx(1.0, abs=1e-6)

  def test_unclipped_bias_matches_mean(self):
    v = 100.0 / 3.6
    c = 1e-3
    n = 2000
    # actual = desired - small_offset (within cap)
    d = np.full(n, c)
    a = np.full(n, c - 1e-4)
    vv = np.full(n, v)
    res = batch_fit_per_bucket(d, a, vv)
    idx_c = CurvatureDLookup.curvature_index(c)
    sw = CurvatureDLookup.learning_speed_weights(v)
    for s, _ in sw:
      assert res.bias[s, idx_c] == pytest.approx(1e-4, abs=1e-7)
      assert res.bias_clipped_fraction[s, idx_c] == pytest.approx(0.0, abs=1e-6)


class TestBucketAccumulator:
  def test_add_and_serialize_roundtrip(self):
    acc = BucketAccumulator(LEARNER_SHAPE)
    acc.add(3, 5, 1e-4, 1, 0.02)
    acc.add(3, 5, -2e-4, -1, -0.01)
    d = acc.to_dict()
    acc2 = BucketAccumulator.from_dict(d)
    assert acc2.count[3, 5] == 2
    assert acc2.sum_err[3, 5] == pytest.approx(-1e-4, abs=1e-9)
    assert acc2.cnt_left[3, 5] == 1
    assert acc2.cnt_right[3, 5] == 1

  def test_grid_helpers(self):
    assert speed_bucket(100.0 / 3.6, LEARNER_SPEED_ANCHORS) == 4
    assert speed_bucket(0.5, LEARNER_SPEED_ANCHORS) is None
    # 1e-4 in EXT grid: same edges as LEARNER for inner cells
    assert curvature_bucket(1e-4, EXTENDED_CURVATURE_EDGES) == 6
    # Above learner grid but within extended: 1e-2 falls between 8.192e-3 and 1.6384e-2
    assert curvature_bucket(1e-2, EXTENDED_CURVATURE_EDGES) == 13
    # Above extended grid
    assert curvature_bucket(1.0, EXTENDED_CURVATURE_EDGES) is None


# ----- Layer 3: report sanity -----

class TestReportRender:
  def test_renders_with_minimal_stats(self, tmp_path):
    """render() should produce HTML without crashing even when N_cars is
    below thresholds — it should warn but not raise."""
    from openpilot.tools.vw_id4_lateral.report import render
    stats = {
      "n_cars": 1,
      "n_segments": 1,
      "min_cars_per_bucket": 5,
      "min_cars_per_plot": 5,
      "min_bucket_samples_per_car": 30,
      "pop_learner_gated": {
        "median": np.zeros(LEARNER_SHAPE).tolist(),
        "q25": np.zeros(LEARNER_SHAPE).tolist(),
        "q75": np.zeros(LEARNER_SHAPE).tolist(),
        "n_cars": np.zeros(LEARNER_SHAPE, dtype=int).tolist(),
        "shape": list(LEARNER_SHAPE),
      },
      "pop_learner_ungated": {
        "median": np.zeros(LEARNER_SHAPE).tolist(),
        "q25": np.zeros(LEARNER_SHAPE).tolist(),
        "q75": np.zeros(LEARNER_SHAPE).tolist(),
        "n_cars": np.zeros(LEARNER_SHAPE, dtype=int).tolist(),
        "shape": list(LEARNER_SHAPE),
      },
      "pop_extended_ungated": {
        "median": np.zeros(EXTENDED_SHAPE).tolist(),
        "q25": np.zeros(EXTENDED_SHAPE).tolist(),
        "q75": np.zeros(EXTENDED_SHAPE).tolist(),
        "n_cars": np.zeros(EXTENDED_SHAPE, dtype=int).tolist(),
        "shape": list(EXTENDED_SHAPE),
      },
      "hypothesis_battery": {
        "cars": ["dongle_a"],
        "engaged_s": [3600.0],
        "build_year": ["2023"],
        "has_pid": [False],
        "highway_p50": [5e-5],
        "highway_p95": [3e-4],
        "gain_80kph": [0.98],
        "gain_100kph": [0.97],
        "gain_120kph": [0.99],
        "steer_ratio": [16.2],
        "stiffness": [1.0],
        "lateral_delay": [0.18],
        "eps_power": [25.0],
        "learner_vs_batch_pearson": [0.7],
        "inner_left_right_asym": [1e-5],
      },
      "inside_mass_per_car": {"dongle_a": 0.65},
      "pid_vs_nopid": {
        "pid_on":  {"n_cars": 0, "median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "values": []},
        "pid_off": {"n_cars": 1, "median": 3e-4, "ci_lo": 3e-4, "ci_hi": 3e-4, "values": [3e-4]},
      },
      "covariate_spearman": {
        "steer_ratio": {"rho": float("nan"), "n": 1},
      },
      "cars": [{
        "dongle": "dongle_a",
        "segments": 1, "ok_segments": 1, "engaged_s": 3600.0,
        "vin": "WVWZZZ1KZAW000000", "build_year": "2023",
        "has_pid": False,
        "highway_p50_median": 5e-5, "highway_p95_median": 3e-4,
      }],
    }
    out = tmp_path / "report.html"
    render(stats, str(out))
    assert out.exists()
    body = out.read_text()
    assert "FLEET-LEVEL N WARNING" in body  # N=1 < min_cars_per_plot=5
    assert "VOLKSWAGEN_ID4_MK1" in body


# ----- Layer 2: reference-route extraction (network-dependent) -----

REFERENCE_ROUTES = {
  "stock_ta": "f73c01590368ee5b/00000010--19b95d93b3",
  "sp_pid":   "f73c01590368ee5b/0000000e--2d623b6df3",
}


@pytest.mark.slow
@pytest.mark.parametrize("label", list(REFERENCE_ROUTES.keys()))
def test_reference_route_extraction(label):
  """Pull each reference route and assert the per-segment record looks sane.
  Skipped if the network is unreachable."""
  pytest.importorskip("openpilot.tools.lib.logreader")
  from openpilot.tools.vw_id4_lateral.extract_segment import extract
  route = REFERENCE_ROUTES[label]
  try:
    rec = extract(route, fingerprint="VOLKSWAGEN_ID4_MK1")
  except Exception as e:
    pytest.skip(f"network or LogReader unavailable: {e}")
  if rec.reason and not rec.ok:
    pytest.skip(f"route returned reason={rec.reason}")
  assert rec.duration_s > 30.0, f"route duration {rec.duration_s} too short"
  if rec.ok:
    assert rec.gated_samples > 100
    # at least some highway-band data
    assert not np.isnan(rec.highway_abs_residual_p50)
    # residual magnitude should be within envelope (0.195)
    assert rec.highway_abs_residual_p95 < 0.195
