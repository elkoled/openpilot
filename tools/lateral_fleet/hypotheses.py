"""Alternative-hypothesis evaluators.

Each hypothesis models an additive correction `delta_c` added to commanded
curvature. Post-correction residual is `resid_after = resid_before - delta_c`.

The evaluator reports both pooled RMS and per-held-out-dongle RMS,
alongside Null. Held-out splits are by dongle (not route, not sample).
A correction whose implied gain K is outside (0, 1] is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from openpilot.tools.lateral_fleet import features

RESID_DEFINITIONS = ('yaw', 'eps')
KEY_COLUMNS = ['speed_idx', 'curv_idx', 'sign']


@dataclass
class HypothesisResult:
  name: str
  resid_kind: str
  rms_before: float
  rms_after: float
  per_dongle: pd.DataFrame
  correction_table: pd.DataFrame = field(default_factory=pd.DataFrame)
  notes: str = ''


def _resid_col(kind: str) -> str:
  return f'resid_{kind}_mean'


def _trivial_result(name: str, resid_kind: str, note: str) -> HypothesisResult:
  return HypothesisResult(name, resid_kind, float('nan'), float('nan'),
                          pd.DataFrame(), notes=note)


def hyp_null(pooled_by_dongle, dongle_meta, resid_kind: str) -> HypothesisResult:
  rows = []
  rcol = _resid_col(resid_kind)
  squared = []
  for dongle, pooled in pooled_by_dongle.items():
    p = pooled[pooled['sufficient']]
    if p.empty:
      continue
    r = p[rcol].to_numpy()
    r = r[np.isfinite(r)]
    if r.size == 0:
      continue
    rms = float(np.sqrt(np.mean(r ** 2)))
    rows.append({'dongle_id': dongle, 'rms_before': rms, 'rms_after': rms, 'n_buckets': int(r.size)})
    squared.append(r ** 2)
  per = pd.DataFrame(rows)
  pooled_sq = np.concatenate(squared) if squared else np.array([0.0])
  rms = float(np.sqrt(np.mean(pooled_sq)))
  return HypothesisResult('null', resid_kind, rms, rms, per, notes='delta_c == 0')


def hyp_per_speed_global(pooled_by_dongle, dongle_meta, resid_kind: str, k_folds: int = 5):
  rcol = _resid_col(resid_kind)
  dongles = sorted(pooled_by_dongle.keys())
  if len(dongles) < 2:
    return _trivial_result('per_speed_global', resid_kind, 'fewer than 2 dongles; cannot cross-validate')
  rng = np.random.default_rng(7)
  fold_assign = rng.permutation(len(dongles)) % k_folds
  fold_of = dict(zip(dongles, fold_assign))
  before_all, after_all, per_rows = [], [], []
  for fold in range(k_folds):
    train_d = [d for d in dongles if fold_of[d] != fold]
    test_d = [d for d in dongles if fold_of[d] == fold]
    if not train_d or not test_d:
      continue
    train = pd.concat([pooled_by_dongle[d][pooled_by_dongle[d]['sufficient']] for d in train_d],
                      ignore_index=True)
    if train.empty:
      continue
    # Hierarchical-style: per-dongle per-speed mean first, then average across train dongles.
    speed_corr: dict[int, float] = {}
    for s_idx, grp in train.groupby('speed_idx'):
      per_dongle_means = []
      for d, g in grp.groupby(grp.index // len(train_d)):  # not a real per-dongle grouping
        pass
      # Compute per-dongle mean residual at this speed by re-grouping on dongle.
      # We carry dongle_id around the train concat via index alignment below.
    # Simpler path: explicitly per-dongle.
    speed_corr = {}
    for s_idx in range(features.NUM_SPEED_ANCHORS):
      per_d_means = []
      for d in train_d:
        p = pooled_by_dongle[d]
        p = p[(p['sufficient']) & (p['speed_idx'] == s_idx)]
        if p.empty:
          continue
        r = p[rcol].to_numpy()
        r = r[np.isfinite(r)]
        if r.size == 0:
          continue
        per_d_means.append(float(np.mean(r)))
      speed_corr[s_idx] = float(np.mean(per_d_means)) if per_d_means else 0.0
    for d in test_d:
      p = pooled_by_dongle[d][pooled_by_dongle[d]['sufficient']]
      if p.empty:
        continue
      delta = p['speed_idx'].map(speed_corr).fillna(0.0).to_numpy()
      r = p[rcol].to_numpy()
      ok = np.isfinite(r)
      r = r[ok]; delta = delta[ok]
      after = r - delta
      before_all.append(r); after_all.append(after)
      per_rows.append({
        'dongle_id': d,
        'rms_before': float(np.sqrt(np.mean(r ** 2))),
        'rms_after': float(np.sqrt(np.mean(after ** 2))),
        'n_buckets': int(r.size),
      })
  if not before_all:
    return _trivial_result('per_speed_global', resid_kind, 'no train/test pairs produced data')
  before = np.concatenate(before_all)
  after = np.concatenate(after_all)
  return HypothesisResult(
    'per_speed_global', resid_kind,
    rms_before=float(np.sqrt(np.mean(before ** 2))),
    rms_after=float(np.sqrt(np.mean(after ** 2))),
    per_dongle=pd.DataFrame(per_rows),
    notes=f'{k_folds}-fold by dongle; per-speed unweighted-mean residual across train dongles',
  )


def hyp_per_vin_scalar(pooled_by_dongle, dongle_meta, resid_kind: str):
  rcol = _resid_col(resid_kind)
  rows = []
  before_all, after_all = [], []
  vin_map = dict(zip(dongle_meta['dongle_id'], dongle_meta['vin'])) if not dongle_meta.empty else {}
  for dongle, pooled in pooled_by_dongle.items():
    p = pooled[pooled['sufficient']]
    if p.empty:
      continue
    r = p[rcol].to_numpy()
    r = r[np.isfinite(r)]
    if r.size == 0:
      continue
    iqr = p[f'resid_{resid_kind}_iqr'].to_numpy()
    iqr = iqr[np.isfinite(iqr)]
    after = (iqr / 1.349) if iqr.size else np.zeros_like(r)
    n = min(r.size, after.size)
    r = r[:n]; after = after[:n]
    before_all.append(r); after_all.append(after)
    rows.append({
      'dongle_id': dongle, 'vin': vin_map.get(dongle, ''),
      'rms_before': float(np.sqrt(np.mean(r ** 2))),
      'rms_after': float(np.sqrt(np.mean(after ** 2))),
      'n_buckets': int(n),
    })
  if not before_all:
    return _trivial_result('per_vin_scalar', resid_kind, 'no sufficient buckets')
  before = np.concatenate(before_all)
  after = np.concatenate(after_all)
  return HypothesisResult(
    'per_vin_scalar', resid_kind,
    rms_before=float(np.sqrt(np.mean(before ** 2))),
    rms_after=float(np.sqrt(np.mean(after ** 2))),
    per_dongle=pd.DataFrame(rows),
    notes='IN-SAMPLE upper bound (per-VIN bias removed; residual is within-bucket spread)',
  )


def hyp_per_fingerprint(pooled_by_dongle, dongle_meta, resid_kind: str):
  if dongle_meta.empty:
    return _trivial_result('per_fingerprint', resid_kind, 'no dongle metadata')
  fps = set(dongle_meta['fingerprint'].dropna().unique())
  if len(fps) <= 1:
    res = hyp_per_speed_global(pooled_by_dongle, dongle_meta, resid_kind)
    return HypothesisResult(
      'per_fingerprint', resid_kind, res.rms_before, res.rms_after,
      res.per_dongle, notes=f'only one fingerprint ({fps}); reduces to per_speed_global')
  return _trivial_result('per_fingerprint', resid_kind,
                         'multi-fingerprint case not exercised in this fleet')


def hyp_lp_conditional(pooled_by_dongle, dongle_meta, resid_kind: str, k_folds: int = 5):
  rcol = _resid_col(resid_kind)
  if dongle_meta.empty:
    return _trivial_result('lp_conditional', resid_kind, 'no dongle metadata')
  sr_map = dict(zip(dongle_meta['dongle_id'], dongle_meta['steer_ratio_mean']))
  dongles = [d for d in pooled_by_dongle if d in sr_map and np.isfinite(sr_map.get(d, np.nan))]
  if len(dongles) < 3:
    return _trivial_result('lp_conditional', resid_kind,
                           'need ≥3 dongles with valid liveParameters')
  rng = np.random.default_rng(11)
  fold_assign = rng.permutation(len(dongles)) % k_folds
  fold_of = dict(zip(dongles, fold_assign))
  before_all, after_all, per_rows = [], [], []
  for fold in range(k_folds):
    train_d = [d for d in dongles if fold_of[d] != fold]
    test_d = [d for d in dongles if fold_of[d] == fold]
    if not train_d or not test_d:
      continue
    sr_train = np.array([sr_map[d] for d in train_d])
    sr_center = float(np.mean(sr_train))
    speed_fit: dict[int, tuple[float, float]] = {}
    for s_idx in range(features.NUM_SPEED_ANCHORS):
      xs, ys, ws = [], [], []
      for d in train_d:
        p = pooled_by_dongle[d]
        p = p[(p['sufficient']) & (p['speed_idx'] == s_idx)]
        if p.empty:
          continue
        r = p[rcol].to_numpy()
        w = p['count'].to_numpy().astype(np.float64)
        ok = np.isfinite(r) & (w > 0)
        if ok.sum() == 0:
          continue
        y = float(np.sum(r[ok] * w[ok]) / np.sum(w[ok]))
        xs.append(sr_map[d] - sr_center); ys.append(y); ws.append(float(np.sum(w[ok])))
      if len(xs) < 2:
        speed_fit[s_idx] = (float(np.mean(ys)) if ys else 0.0, 0.0)
        continue
      xs_a = np.array(xs); ys_a = np.array(ys); ws_a = np.array(ws)
      X = np.vstack([np.ones_like(xs_a), xs_a]).T
      W = np.diag(ws_a)
      try:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ ys_a)
        speed_fit[s_idx] = (float(beta[0]), float(beta[1]))
      except np.linalg.LinAlgError:
        speed_fit[s_idx] = (float(np.average(ys_a, weights=ws_a)), 0.0)
    for d in test_d:
      p = pooled_by_dongle[d][pooled_by_dongle[d]['sufficient']]
      if p.empty:
        continue
      dx = sr_map[d] - sr_center
      delta = np.array([speed_fit.get(int(s), (0.0, 0.0))[0] + speed_fit.get(int(s), (0.0, 0.0))[1] * dx
                        for s in p['speed_idx']])
      r = p[rcol].to_numpy()
      ok = np.isfinite(r); r = r[ok]; delta = delta[ok]
      after = r - delta
      before_all.append(r); after_all.append(after)
      per_rows.append({
        'dongle_id': d, 'steer_ratio_mean': sr_map[d],
        'rms_before': float(np.sqrt(np.mean(r ** 2))),
        'rms_after': float(np.sqrt(np.mean(after ** 2))),
        'n_buckets': int(r.size),
      })
  if not before_all:
    return _trivial_result('lp_conditional', resid_kind, 'no fold produced data')
  before = np.concatenate(before_all)
  after = np.concatenate(after_all)
  return HypothesisResult(
    'lp_conditional', resid_kind,
    rms_before=float(np.sqrt(np.mean(before ** 2))),
    rms_after=float(np.sqrt(np.mean(after ** 2))),
    per_dongle=pd.DataFrame(per_rows),
    notes=f'{k_folds}-fold by dongle; per-speed linear in (steerRatio - mean)',
  )


def hyp_sunnypilot_learner(pooled_by_dongle, dongle_meta, resid_kind: str):
  return _trivial_result(
    'sunnypilot_learner', resid_kind,
    'not implemented in this cut: requires vendoring CurvatureEstimator from '
    '~/openpilot10:selfdrive/locationd/curvatured.py and a per-route replay driver.',
  )


def hyp_batch_per_dongle_bucket(pooled_by_dongle, dongle_meta, resid_kind: str):
  rcol = _resid_col(resid_kind)
  before_all, after_all, per_rows = [], [], []
  quarantined_total = 0
  for dongle, pooled in pooled_by_dongle.items():
    p = pooled[pooled['sufficient']].copy()
    if p.empty:
      continue
    r = p[rcol].to_numpy()
    gain_col = f'gain_{resid_kind}_mean'
    K = p[gain_col].to_numpy() if gain_col in p.columns else np.full_like(r, np.nan)
    plausible = np.isfinite(K) & (K > 0) & (K <= 1.0)
    delta = np.where(plausible, r, 0.0)
    quarantined_total += int((~plausible).sum())
    after = r - delta
    ok = np.isfinite(r)
    r = r[ok]; after = after[ok]
    if r.size == 0:
      continue
    before_all.append(r); after_all.append(after)
    per_rows.append({
      'dongle_id': dongle,
      'rms_before': float(np.sqrt(np.mean(r ** 2))),
      'rms_after': float(np.sqrt(np.mean(after ** 2))),
      'n_buckets': int(r.size),
      'n_quarantined': int((~plausible).sum()),
    })
  if not before_all:
    return _trivial_result('batch_per_dongle_bucket', resid_kind, 'no sufficient buckets')
  before = np.concatenate(before_all)
  after = np.concatenate(after_all)
  return HypothesisResult(
    'batch_per_dongle_bucket', resid_kind,
    rms_before=float(np.sqrt(np.mean(before ** 2))),
    rms_after=float(np.sqrt(np.mean(after ** 2))),
    per_dongle=pd.DataFrame(per_rows),
    notes=f'IN-SAMPLE ceiling; {quarantined_total} buckets quarantined (K∉(0,1])',
  )


HYPOTHESES: dict[str, Callable] = {
  'null': hyp_null,
  'per_speed_global': hyp_per_speed_global,
  'per_vin_scalar': hyp_per_vin_scalar,
  'per_fingerprint': hyp_per_fingerprint,
  'lp_conditional': hyp_lp_conditional,
  'sunnypilot_learner': hyp_sunnypilot_learner,
  'batch_per_dongle_bucket': hyp_batch_per_dongle_bucket,
}


def evaluate_all(pooled_by_dongle, dongle_meta) -> pd.DataFrame:
  rows = []
  for name, fn in HYPOTHESES.items():
    for kind in RESID_DEFINITIONS:
      try:
        res = fn(pooled_by_dongle, dongle_meta, kind)
      except Exception as e:  # noqa: BLE001
        rows.append({'hypothesis': name, 'resid_kind': kind,
                     'rms_before': float('nan'), 'rms_after': float('nan'),
                     'n_dongles': 0, 'notes': f'error: {type(e).__name__}: {e}'})
        continue
      rows.append({
        'hypothesis': name, 'resid_kind': kind,
        'rms_before': res.rms_before, 'rms_after': res.rms_after,
        'n_dongles': int(res.per_dongle.shape[0]),
        'notes': res.notes,
      })
  return pd.DataFrame(rows)
