"""CLI: `python -m tools.lateral_fleet ...`"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

from openpilot.tools.lateral_fleet import aggregate, cache, hypotheses, ingest, plant_fit, report


def _read_routes_csv(path: Path) -> list[tuple[str, str]]:
  routes: list[tuple[str, str]] = []
  with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      d = (row.get('dongle_id') or row.get('dongle') or '').strip()
      r = (row.get('route_id') or row.get('route') or '').strip()
      if d and r:
        routes.append((d, r))
  return routes


def _read_segments_csv(path: Path, min_engaged: float = 0.3) -> list[tuple[str, str, str]]:
  """Read the user's eps_seglist.csv. Returns (dongle, segment_id, rlog_url).

  The `segment` column is in canonical form <dongle>/<route_log>--<seg>.
  We use the portion after the slash (i.e. <route_log>--<seg>) as the cache
  key, so cache layout is `<dongle>/<route_log>--<seg>.parquet`.
  """
  out: list[tuple[str, str, str]] = []
  with path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      try:
        ef = float(row.get('engaged_frac', '0') or 0)
      except ValueError:
        ef = 0.0
      if ef < min_engaged:
        continue
      dongle = (row.get('dongle') or '').strip()
      segment = (row.get('segment') or '').strip()
      url = (row.get('rlog_url') or '').strip()
      if not dongle or not segment or not url:
        continue
      if '/' in segment:
        seg_id = segment.split('/', 1)[1]
      else:
        seg_id = segment
      out.append((dongle, seg_id, url))
  return out


def _process_one(args_tuple):
  d, r, retry_failed = args_tuple
  if not retry_failed and cache.already_ok(d, r):
    return cache.read_status(d, r)
  return ingest.process_route(d, r)


def _process_one_url(args_tuple):
  d, s, url, retry_failed = args_tuple
  if not retry_failed and cache.already_ok(d, s):
    return cache.read_status(d, s)
  return ingest.process_url(d, s, url)


def cmd_ingest(args: argparse.Namespace) -> int:
  routes = _read_routes_csv(Path(args.routes))
  if args.limit:
    routes = routes[: args.limit]
  if not routes:
    print('no routes in CSV', file=sys.stderr)
    return 2
  print(f'ingesting {len(routes)} routes with {args.workers} workers ...')
  t0 = time.time()
  with mp.get_context('spawn').Pool(args.workers) as pool:
    counts: dict[str, int] = {}
    for i, rs in enumerate(pool.imap_unordered(
        _process_one, [(d, r, args.retry_failed) for d, r in routes]), start=1):
      counts[rs.status] = counts.get(rs.status, 0) + 1
      if i % 50 == 0 or i == len(routes):
        elapsed = time.time() - t0
        print(f'  [{i}/{len(routes)}] elapsed={elapsed:.0f}s counts={counts}', flush=True)
  print(f'done in {time.time() - t0:.0f}s; counts={counts}')
  return 0


def cmd_ingest_csv(args: argparse.Namespace) -> int:
  """Ingest the user's eps_seglist.csv (segment + rlog_url + engaged_frac)."""
  segs = _read_segments_csv(Path(args.csv), min_engaged=args.min_engaged)
  if args.limit:
    segs = segs[: args.limit]
  if not segs:
    print('no segments survive filter', file=sys.stderr)
    return 2
  print(f'ingesting {len(segs)} segments with {args.workers} workers '
        f'(engaged_frac >= {args.min_engaged}) ...')
  t0 = time.time()
  with mp.get_context('spawn').Pool(args.workers) as pool:
    counts: dict[str, int] = {}
    for i, rs in enumerate(pool.imap_unordered(
        _process_one_url, [(d, s, u, args.retry_failed) for d, s, u in segs]), start=1):
      counts[rs.status] = counts.get(rs.status, 0) + 1
      if i % 25 == 0 or i == len(segs):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 1e-3)
        eta = (len(segs) - i) / max(rate, 1e-3)
        print(f'  [{i}/{len(segs)}] elapsed={elapsed:.0f}s rate={rate:.1f}/s '
              f'eta={eta:.0f}s counts={counts}', flush=True)
  print(f'done in {time.time() - t0:.0f}s; counts={counts}')
  return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
  s = aggregate.aggregate_run(Path(args.run), n_bootstrap=args.bootstrap,
                              min_dongles=args.min_dongles, hierarchical=not args.weighted)
  print(f'aggregate complete: {s}')
  return 0


def cmd_plant(args: argparse.Namespace) -> int:
  for kind in ('yaw', 'eps'):
    print(f'fitting plant ({kind}) ...')
    s = plant_fit.fit_all(Path(args.run), resid_kind=kind)
    print(f'  {s}')
  return 0


def cmd_report(args: argparse.Namespace) -> int:
  out = report.build_report(Path(args.run), min_dongles=args.min_dongles)
  print(f'report: {out}')
  return 0


def cmd_hypothesis(args: argparse.Namespace) -> int:
  pooled_dir = Path(args.run) / 'dongle_pooled'
  if not pooled_dir.exists():
    print(f'no aggregate run at {pooled_dir}', file=sys.stderr)
    return 2
  import pandas as pd
  pooled = {f.stem: pd.read_parquet(f) for f in pooled_dir.glob('*.parquet')}
  meta = pd.read_parquet(Path(args.run) / 'dongles.parquet') \
    if (Path(args.run) / 'dongles.parquet').exists() else pd.DataFrame()
  fn = hypotheses.HYPOTHESES.get(args.name)
  if fn is None:
    print(f'unknown hypothesis {args.name}', file=sys.stderr)
    return 2
  for kind in hypotheses.RESID_DEFINITIONS:
    res = fn(pooled, meta, kind)
    print(f'\n=== {res.name} ({res.resid_kind}) ===')
    print(f'rms_before = {res.rms_before:.6g}')
    print(f'rms_after  = {res.rms_after:.6g}')
    print(f'notes      = {res.notes}')
    if not res.per_dongle.empty:
      print(res.per_dongle.to_string(index=False))
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog='tools.lateral_fleet')
  sub = ap.add_subparsers(dest='cmd', required=True)

  ap_i = sub.add_parser('ingest')
  ap_i.add_argument('--routes', required=True)
  ap_i.add_argument('--workers', type=int, default=8)
  ap_i.add_argument('--retry-failed', action='store_true')
  ap_i.add_argument('--limit', type=int, default=0)
  ap_i.set_defaults(func=cmd_ingest)

  ap_ic = sub.add_parser('ingest-csv')
  ap_ic.add_argument('--csv', required=True)
  ap_ic.add_argument('--workers', type=int, default=8)
  ap_ic.add_argument('--retry-failed', action='store_true')
  ap_ic.add_argument('--limit', type=int, default=0)
  ap_ic.add_argument('--min-engaged', type=float, default=0.3)
  ap_ic.set_defaults(func=cmd_ingest_csv)

  ap_a = sub.add_parser('aggregate')
  ap_a.add_argument('--run', required=True)
  ap_a.add_argument('--bootstrap', type=int, default=aggregate.DEFAULT_BOOTSTRAP)
  ap_a.add_argument('--min-dongles', type=int, default=aggregate.DEFAULT_MIN_DONGLES)
  ap_a.add_argument('--weighted', action='store_true',
                    help='use count-weighted pooling instead of hierarchical')
  ap_a.set_defaults(func=cmd_aggregate)

  ap_p = sub.add_parser('plant')
  ap_p.add_argument('--run', required=True)
  ap_p.set_defaults(func=cmd_plant)

  ap_r = sub.add_parser('report')
  ap_r.add_argument('--run', required=True)
  ap_r.add_argument('--min-dongles', type=int, default=aggregate.DEFAULT_MIN_DONGLES)
  ap_r.set_defaults(func=cmd_report)

  ap_h = sub.add_parser('hypothesis')
  ap_h.add_argument('--run', required=True)
  ap_h.add_argument('--name', required=True, choices=sorted(hypotheses.HYPOTHESES.keys()))
  ap_h.set_defaults(func=cmd_hypothesis)

  args = ap.parse_args(argv)
  return int(args.func(args) or 0)


if __name__ == '__main__':
  sys.exit(main())
