"""Smoke test on the sunnypilot-PID reference route. Slow (~30s) -- gated by env var.

Marked slow because LogReader hits the network; opt-in via FLEET_SMOKE=1.
"""
import os

import numpy as np
import pytest

from tools.lateral_maneuvers.id4_fleet.extract import run as extract_run


pytestmark = pytest.mark.skipif(
  os.environ.get("FLEET_SMOKE", "0") != "1",
  reason="set FLEET_SMOKE=1 to run network-dependent smoke test",
)


def test_pid_reference_route_extracts_meaningful_features():
  res = extract_run("f73c01590368ee5b", "0000000e--2d623b6df3", "pid_reference")
  assert res.status == "ok", res.error
  assert res.car_fingerprint == "VOLKSWAGEN_ID4_MK1"
  assert res.engaged_seconds > 60
  assert res.n_samples_gated > 100

  f = res.features
  # P1 bucket counts populated
  c = f["P1_desired_yaw_count"]
  assert c.sum() > 100
  # residuals are in the expected order of magnitude for highway-gentle steering
  # (a few times 1e-4 rad/m).
  sq = f["P1_desired_yaw_sumsq_residual"]
  valid = c > 0
  rms_per_bucket = np.where(valid, np.sqrt(sq / np.where(valid, c, 1.0)), 0.0)
  max_rms = float(rms_per_bucket.max())
  assert 1e-5 < max_rms < 5e-3, f"unexpected residual magnitude max_rms={max_rms:.2e}"

  # conditioning fields populated
  assert f["lateral_delay_mean"] > 0.0
  assert f["steer_ratio_mean"] > 10.0
  assert f["stiffness_mean"] > 0.5

  # xcorr at zero or small positive lag with high correlation
  r = f["P1_desired_yaw_xcorr_r"]
  assert np.nanmax(r) > 0.8
