#!/usr/bin/env python3
"""
Deploy-time EPS correction from a fitted plant model.

Reads plant_fit.json (output of plant_fit.py) and provides:

  EpsCorrection.from_json(path)
      .corrected_curvature(desired, v_ego)  -> float
      .corrected_curvature_with_confidence(desired, v_ego)
          -> (corrected, in_supported_region: bool)

The static-model inverse is used for application:
    actual ≈ G · desired + bias  →  corrected_desired = (target - bias) / G

For (car, speed_bin) cells flagged not-deployable the correction passes
through unchanged. Speed interpolation is linear between adjacent anchors;
outside the anchor range the correction fades to identity (smoothstep) so
the controller never sees a step discontinuity at the deployment boundary.

The choice to apply the *static* inverse (not the ARX) is deliberate:
- Steady-state highway tracking is what we're correcting.
- ARX phase-lead pre-compensation requires accurate τ — which our
  data-limited fits don't justify deploying.
- The static inverse cannot induce oscillation; the worst case is
  passthrough.

Usage in a controller (sketch):

    eps = EpsCorrection.from_json("plant_fit.json", dongle=THIS_DONGLE)
    corrected_curvature = eps.corrected_curvature(actuators.curvature, CS.vEgo)
    apply_curvature = corrected_curvature + (CS.curvature_meas - CC.currentCurvature)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

SPEED_ANCHORS_KPH = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
FADE_BAND_KPH = 5.0  # smoothstep fade-in width below the lowest deployable anchor


@dataclass
class _PerSpeedFit:
  v_anchor: float        # m/s
  G: float
  bias: float
  Td_s: float
  deployable: bool
  loo_rmse: float
  in_sample_r2: float


class EpsCorrection:
  """Per-dongle speed-interpolated static EPS inverse.

  Constructed via from_json() with an optional dongle filter. Cells flagged
  not-deployable are skipped — within the supported speed range we
  interpolate between adjacent deployable anchors; outside, we smoothstep
  to identity (G=1, bias=0)."""

  def __init__(self, dongle: str, fits: list[_PerSpeedFit]):
    self.dongle = dongle
    self.fits = sorted(fits, key=lambda f: f.v_anchor)

  @classmethod
  def from_json(cls, path: str, dongle: str | None = None) -> "EpsCorrection":
    with open(path) as f:
      data = json.load(f)
    if dongle is None:
      if len(data) == 1:
        dongle = next(iter(data))
      else:
        raise ValueError(f"plant_fit.json has multiple dongles {list(data)}; "
                         "specify dongle=")
    if dongle not in data:
      raise KeyError(f"dongle {dongle} not in plant_fit.json (have {list(data)})")
    bins = data[dongle]
    fits: list[_PerSpeedFit] = []
    for bin_name, b in bins.items():
      v_anchor = float(bin_name.replace("kph", "")) / 3.6
      # Prefer the chosen model's static parameters; if ARX was chosen we
      # still expose its static-equivalent (G_static = K_arx, bias_static = 0
      # because ARX absorbs DC gain into K).
      if b.get("chosen_model") == "arx":
        G = b.get("arx_K") or float("nan")
        bias = 0.0
        Td_s = b.get("arx_Td_s") or 0.0
        loo_rmse = b.get("arx_loo_rmse") or float("nan")
        r2 = b.get("arx_in_sample_r2") or float("nan")
      else:
        G = b.get("static_G") or float("nan")
        bias = b.get("static_bias") or 0.0
        Td_s = b.get("static_Td_s") or 0.0
        loo_rmse = b.get("static_loo_rmse") or float("nan")
        r2 = b.get("static_in_sample_r2") or float("nan")
      fits.append(_PerSpeedFit(
        v_anchor=v_anchor, G=float(G if G is not None else float("nan")),
        bias=float(bias if bias is not None else 0.0),
        Td_s=float(Td_s if Td_s is not None else 0.0),
        deployable=bool(b.get("deployable", False)),
        loo_rmse=float(loo_rmse if loo_rmse is not None else float("nan")),
        in_sample_r2=float(r2 if r2 is not None else float("nan")),
      ))
    return cls(dongle, fits)

  def _deployable_anchors(self) -> list[_PerSpeedFit]:
    return [f for f in self.fits if f.deployable and 0.5 < f.G < 1.5
            and math.isfinite(f.G) and math.isfinite(f.bias)]

  def in_supported_region(self, v_ego: float) -> bool:
    dep = self._deployable_anchors()
    if not dep:
      return False
    v_lo = dep[0].v_anchor - FADE_BAND_KPH / 3.6
    v_hi = dep[-1].v_anchor + FADE_BAND_KPH / 3.6
    return v_lo <= float(v_ego) <= v_hi

  def _interp_gain_bias(self, v_ego: float) -> tuple[float, float, float]:
    """Returns (G_eff, bias_eff, alpha_in_region) where alpha is the
    smoothstep weighting (1.0 in supported range, fades to 0 outside)."""
    dep = self._deployable_anchors()
    if not dep:
      return 1.0, 0.0, 0.0
    v = float(v_ego)
    if v <= dep[0].v_anchor:
      # below lowest deployable anchor — fade smoothly
      v_fade_lo = dep[0].v_anchor - FADE_BAND_KPH / 3.6
      if v <= v_fade_lo:
        return 1.0, 0.0, 0.0
      alpha = (v - v_fade_lo) / (FADE_BAND_KPH / 3.6)
      alpha = self._smoothstep(alpha)
      G = alpha * dep[0].G + (1.0 - alpha) * 1.0
      b = alpha * dep[0].bias + (1.0 - alpha) * 0.0
      return G, b, alpha
    if v >= dep[-1].v_anchor:
      v_fade_hi = dep[-1].v_anchor + FADE_BAND_KPH / 3.6
      if v >= v_fade_hi:
        return 1.0, 0.0, 0.0
      alpha = 1.0 - (v - dep[-1].v_anchor) / (FADE_BAND_KPH / 3.6)
      alpha = self._smoothstep(max(0.0, alpha))
      G = alpha * dep[-1].G + (1.0 - alpha) * 1.0
      b = alpha * dep[-1].bias + (1.0 - alpha) * 0.0
      return G, b, alpha
    # interpolate between adjacent anchors
    for i in range(len(dep) - 1):
      if dep[i].v_anchor <= v <= dep[i + 1].v_anchor:
        span = dep[i + 1].v_anchor - dep[i].v_anchor
        t = (v - dep[i].v_anchor) / max(span, 1e-6)
        G = (1.0 - t) * dep[i].G + t * dep[i + 1].G
        b = (1.0 - t) * dep[i].bias + t * dep[i + 1].bias
        return G, b, 1.0
    return 1.0, 0.0, 0.0

  @staticmethod
  def _smoothstep(x: float) -> float:
    y = float(np.clip(x, 0.0, 1.0))
    return y * y * (3.0 - 2.0 * y)

  def corrected_curvature(self, desired_curvature: float, v_ego: float) -> float:
    """Apply (target - bias) / G inverse correction with bounded G and
    out-of-region passthrough."""
    G, b, alpha = self._interp_gain_bias(v_ego)
    if alpha <= 0.0 or not (0.5 < G < 1.5):
      return float(desired_curvature)
    return float((float(desired_curvature) - b) / G)

  def corrected_curvature_with_confidence(self, desired_curvature: float,
                                          v_ego: float) -> tuple[float, bool]:
    return self.corrected_curvature(desired_curvature, v_ego), self.in_supported_region(v_ego)


def main():
  """CLI sanity check: print the correction curve for one dongle."""
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--plant-fit", default="plant_fit.json")
  p.add_argument("--dongle", required=True)
  args = p.parse_args()
  c = EpsCorrection.from_json(args.plant_fit, dongle=args.dongle)
  print(f"dongle={args.dongle}  deployable_anchors={len(c._deployable_anchors())}/{len(c.fits)}")
  for f in c.fits:
    print(f"  {int(f.v_anchor*3.6):3d} km/h  G={f.G:.3f}  bias={f.bias:+.5f}  "
          f"Td={f.Td_s*1000:.0f}ms  R²={f.in_sample_r2:.2f}  "
          f"LOO_rmse={f.loo_rmse:.5f}  deployable={f.deployable}")
  # Sample correction at a few (desired, v) points
  print("\nsample corrections (desired_curvature -> corrected):")
  for v_kph in (40, 80, 100, 120):
    for d in (-5e-3, -1e-3, 0.0, 1e-3, 5e-3):
      out = c.corrected_curvature(d, v_kph / 3.6)
      print(f"  v={v_kph:3d}  d={d:+.4f}  -> {out:+.5f}  (delta={out-d:+.5f})")


if __name__ == "__main__":
  import sys
  sys.exit(main())
