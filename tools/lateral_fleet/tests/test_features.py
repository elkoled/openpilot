from __future__ import annotations

import numpy as np
import pandas as pd

from openpilot.tools.lateral_fleet import features, hypotheses


def test_speed_bucket_picks_nearest_anchor():
  v = np.array([5.0, 5.6, 16.7, 100.0])
  idx = features.speed_bucket(v)
  assert idx[0] == 0 and idx[1] == 0
  assert idx[2] == 2
  assert idx[3] == features.NUM_SPEED_ANCHORS - 1


def test_curvature_bucket_in_range_and_outside():
  c = np.array([5e-7, 1e-6, 3e-6, 1e-4, 5e-3])
  idx = features.curvature_bucket(c)
  assert idx[0] == -1
  assert idx[1] == 0
  assert idx[2] == 1
  assert idx[3] > 0
  assert idx[4] == -1


def test_curvature_bucket_left_right_symmetric():
  c_pos = np.array([1.5e-4, 3.5e-4])
  c_neg = -c_pos
  assert (features.curvature_bucket(c_pos) == features.curvature_bucket(c_neg)).all()


def test_sign_index_basic():
  c = np.array([1e-4, -1e-4, 0.0])
  s = features.sign_index(c)
  assert s[0] == 1 and s[1] == -1 and s[2] == 0


def test_lat_accel_gate_excludes_high_g_turns():
  c = np.array([1e-3, 1e-3])
  v = np.array([20.0, 40.0])
  ok = features.lat_accel_gate(c, v)
  assert ok[0] and not ok[1]


def test_roll_gate_strict_sunnypilot_threshold():
  roll = np.array([0.0, np.deg2rad(0.5), np.deg2rad(2.0)])
  ok = features.roll_gate(roll, threshold=features.MAX_LEARN_ROLL_LAT_ACCEL)
  assert ok[0] and ok[1] and not ok[2]


def test_roll_gate_default_permits_real_road_camber():
  roll = np.array([0.0, np.deg2rad(2.0), np.deg2rad(6.0)])
  ok = features.roll_gate(roll)
  assert ok[0] and ok[1]
  assert not ok[2]


def test_yaw_rate_std_gate_threshold():
  ok = features.yaw_rate_std_gate(np.array([0.0, 0.99, 1.0, 1.5]))
  assert ok[0] and ok[1] and not ok[2] and not ok[3]


def test_engagement_buffer_holds_after_rising_edge():
  n = 300
  lat_active = np.zeros(n, dtype=bool)
  lat_active[100:] = True
  pressed = np.zeros(n, dtype=bool)
  mask = features.engagement_buffer_mask(lat_active, pressed)
  buf = int(features.MIN_ENGAGE_BUFFER_S * features.RESAMPLE_HZ)
  assert not mask[:100 + buf].any()
  assert mask[100 + buf + 1:].all()


def test_engagement_buffer_resets_on_press():
  n = 400
  lat_active = np.ones(n, dtype=bool)
  pressed = np.zeros(n, dtype=bool)
  pressed[200:210] = True
  mask = features.engagement_buffer_mask(lat_active, pressed)
  buf = int(features.MIN_ENGAGE_BUFFER_S * features.RESAMPLE_HZ)
  assert not mask[200:210].any()
  assert not mask[210:210 + buf].any()
  assert mask[-1]


def test_gain_ratio_drops_near_zero_cmd():
  c_cmd = np.array([1e-6, 1e-4, 1e-3])
  c_act = np.array([1e-6, 9e-5, 9e-4])
  g = features.compute_gain_ratio(c_act, c_cmd)
  assert np.isnan(g[0])
  assert np.isclose(g[1], 0.9)
  assert np.isclose(g[2], 0.9)


def test_shift_by_delay_introduces_correct_lag():
  arr = np.arange(20, dtype=float)
  shifted = features.shift_by_delay(arr, delay_s=0.1)
  assert (shifted[:5] == arr[0]).all()
  assert (shifted[5:] == arr[:-5]).all()


def test_residual_sign_convention():
  c_cmd = np.array([1e-3, -1e-3])
  c_yaw = np.array([5e-4, -5e-4])
  resid = c_yaw - c_cmd
  assert resid[0] < 0
  assert resid[1] > 0


def test_batch_hypothesis_quarantines_K_above_one():
  pooled = pd.DataFrame([
    {'speed_idx': 3, 'curv_idx': 5, 'sign': 1, 'count': 1000, 'n_routes': 5, 'sufficient': True,
     'resid_yaw_mean': 1e-4, 'resid_yaw_median': 0, 'resid_yaw_iqr': 0,
     'resid_eps_mean': 0, 'resid_eps_median': 0, 'resid_eps_iqr': 0,
     'gain_yaw_mean': 1.5, 'gain_yaw_median': 1.5, 'gain_eps_mean': 1.5, 'gain_eps_median': 1.5,
     'torque_driver_mean': 0, 'hca_power_mean': 0, 'steer_ratio_mean': 15,
     'stiffness_factor_mean': 1.0, 'lateral_delay_mean': 0.15},
    {'speed_idx': 3, 'curv_idx': 6, 'sign': 1, 'count': 1000, 'n_routes': 5, 'sufficient': True,
     'resid_yaw_mean': 1e-4, 'resid_yaw_median': 0, 'resid_yaw_iqr': 0,
     'resid_eps_mean': 0, 'resid_eps_median': 0, 'resid_eps_iqr': 0,
     'gain_yaw_mean': 0.9, 'gain_yaw_median': 0.9, 'gain_eps_mean': 0.9, 'gain_eps_median': 0.9,
     'torque_driver_mean': 0, 'hca_power_mean': 0, 'steer_ratio_mean': 15,
     'stiffness_factor_mean': 1.0, 'lateral_delay_mean': 0.15},
  ])
  res = hypotheses.hyp_batch_per_dongle_bucket({'dongleX': pooled}, pd.DataFrame(), 'yaw')
  assert res.rms_after > 0
  assert 'quarantined' in res.notes


def test_null_hypothesis_unchanged():
  pooled = pd.DataFrame([
    {'speed_idx': 3, 'curv_idx': 5, 'sign': 1, 'count': 1000, 'n_routes': 5, 'sufficient': True,
     'resid_yaw_mean': 2e-4, 'resid_yaw_median': 0, 'resid_yaw_iqr': 0,
     'resid_eps_mean': 0, 'resid_eps_median': 0, 'resid_eps_iqr': 0,
     'gain_yaw_mean': 0.9, 'gain_yaw_median': 0.9, 'gain_eps_mean': 0.9, 'gain_eps_median': 0.9,
     'torque_driver_mean': 0, 'hca_power_mean': 0, 'steer_ratio_mean': 15,
     'stiffness_factor_mean': 1.0, 'lateral_delay_mean': 0.15},
  ])
  res = hypotheses.hyp_null({'d1': pooled}, pd.DataFrame(), 'yaw')
  assert res.rms_before == res.rms_after
