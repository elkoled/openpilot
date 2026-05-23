"""Per-route artifact storage.

Layout:
  $LATERAL_FLEET_CACHE/<dongle>/<route_or_segment>.parquet            -- timeline
  $LATERAL_FLEET_CACHE/<dongle>/<route_or_segment>.buckets.parquet    -- bucket stats
  $LATERAL_FLEET_CACHE/<dongle>/<route_or_segment>.status.json        -- quarantine sidecar
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd

CACHE_ROOT = Path(os.environ.get('LATERAL_FLEET_CACHE',
                                 Path.home() / '.commacache' / 'lateral_fleet'))

Status = Literal[
  'ok', 'missing', 'qlog_only', 'wrong_fingerprint',
  'corrupted', 'no_can_bus', 'no_engaged_time', 'engaged_frac_too_low',
]


@dataclass
class RouteStatus:
  dongle_id: str
  route_id: str
  status: Status
  message: str = ''
  written_at: float = 0.0
  fingerprint: str = ''
  vin: str = ''
  can_bus: int = -1
  duration_engaged_s: float = 0.0
  duration_strict_gated_s: float = 0.0


def _paths(dongle: str, route: str) -> tuple[Path, Path]:
  base = CACHE_ROOT / dongle
  return base / f'{route}.parquet', base / f'{route}.status.json'


def status_path(dongle: str, route: str) -> Path:
  return _paths(dongle, route)[1]


def parquet_path(dongle: str, route: str) -> Path:
  return _paths(dongle, route)[0]


def read_status(dongle: str, route: str) -> RouteStatus | None:
  p = status_path(dongle, route)
  if not p.exists():
    return None
  try:
    with p.open() as f:
      d = json.load(f)
    return RouteStatus(**d)
  except (json.JSONDecodeError, TypeError):
    return None


def write_status(rs: RouteStatus) -> None:
  p = status_path(rs.dongle_id, rs.route_id)
  p.parent.mkdir(parents=True, exist_ok=True)
  rs.written_at = time.time()
  with p.open('w') as f:
    json.dump(asdict(rs), f, indent=2)


def write_artifact(dongle: str, route: str, timeline: pd.DataFrame, buckets: pd.DataFrame) -> Path:
  p, _ = _paths(dongle, route)
  p.parent.mkdir(parents=True, exist_ok=True)
  timeline.to_parquet(p, compression='zstd', index=False)
  buckets.to_parquet(p.with_suffix('.buckets.parquet'), compression='zstd', index=False)
  return p


def read_timeline(dongle: str, route: str) -> pd.DataFrame:
  return pd.read_parquet(parquet_path(dongle, route))


def read_buckets(dongle: str, route: str) -> pd.DataFrame:
  return pd.read_parquet(parquet_path(dongle, route).with_suffix('.buckets.parquet'))


def already_ok(dongle: str, route: str) -> bool:
  rs = read_status(dongle, route)
  return rs is not None and rs.status == 'ok'
