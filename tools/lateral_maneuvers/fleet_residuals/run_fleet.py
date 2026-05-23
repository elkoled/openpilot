"""Multi-process driver for extract_route across a route list.

Usage:
    python -m tools.lateral_maneuvers.fleet_residuals.run_fleet \\
        --routes routes.csv \\
        --workers 16 \\
        --out-dir tools/lateral_maneuvers/fleet_residuals/out

Route list format - one route per line, comments with `#`:

    dongle_id/route_id[/segment]
    f73c01590368ee5b/0000000e--2d623b6df3
    f73c01590368ee5b/00000010--19b95d93b3/a

If no `/segment` is given, `/a` (auto: rlogs with qlog fallback) is appended.

Resumable: routes whose output pickle already exists are skipped. Delete the
pickle to force a re-extract. Failures land in `<out_dir>/failed.log` with
their category and traceback.

Memory: each worker processes one route, builds the summary, writes the
pickle and discards the working data. Steady-state RAM is bounded by the
worker count, not the route count.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

FAIL_CATEGORIES = {
  'not_found': ('404', 'no such', 'NoSuchKey', 'segment_range', 'route_not_found'),
  'qlog_only_no_can': ('no_can', 'all qlogs', 'no CAN'),
  'wrong_fingerprint': ('mock', 'fingerprint', 'unknown car'),
  'corrupted': ('Premature', 'invalid capnp', 'bad zstd', 'EOF'),
}


def categorise_failure(msg: str) -> str:
  m = (msg or '').lower()
  for cat, tokens in FAIL_CATEGORIES.items():
    for tok in tokens:
      if tok.lower() in m:
        return cat
  return 'other'


def pickle_path_for(route_id: str, out_dir: str) -> str:
  safe = route_id.replace('/', '__').replace('|', '__')
  return os.path.join(out_dir, f'{safe}.pkl')


def parse_route_list(path: str) -> list[tuple[str, str]]:
  """Returns list of (id_for_naming, identifier_for_LogReader).

  Supports two formats:
    1. Plain text: one "dongle/route[/segment]" per line. Comments with '#'.
       LogReader gets the same string.
    2. CSV with header containing 'segment' and 'rlog_url' columns (the
       eps_seglist.csv format). LogReader gets the URL; pickle name uses
       segment.
  """
  with open(path) as f:
    head = f.readline().strip()
  out: list[tuple[str, str]] = []
  if 'segment' in head and 'rlog_url' in head:
    with open(path) as f:
      r = csv.DictReader(f)
      for row in r:
        seg = row.get('segment', '').strip()
        url = row.get('rlog_url', '').strip()
        if seg and url:
          out.append((seg, url))
    return out

  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      if line.count('/') == 1:
        line = line + '/a'
      out.append((line, line))
  return out


def _worker(name: str, identifier: str, out_dir: str, max_seconds: float | None) -> dict:
  # Re-import inside the worker so each process pays the import cost once.
  from openpilot.tools.lateral_maneuvers.fleet_residuals.extract import extract_route

  t0 = time.time()
  try:
    s = extract_route(identifier, out_dir=out_dir, max_seconds=max_seconds,
                      output_name=name)
    return {
      'name': name,
      'status': 'ok' if 'error' not in s else 'extract_error',
      'error': s.get('error'),
      'wall_s': time.time() - t0,
      'learn_samples': s.get('learn_samples', 0),
      'plant_fit_count': len(s.get('plant_fits', [])),
    }
  except Exception as e:
    return {
      'name': name,
      'status': 'crash',
      'error': str(e),
      'traceback': traceback.format_exc(),
      'wall_s': time.time() - t0,
      'category': categorise_failure(str(e)),
    }


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--routes', required=True, help='Path to CSV/text file of routes (one per line, # for comments)')
  ap.add_argument('--out-dir', default='tools/lateral_maneuvers/fleet_residuals/out')
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
  ap.add_argument('--max-seconds', type=float, default=None, help='Per-route processing time cap')
  ap.add_argument('--limit', type=int, default=None, help='Process at most N routes (for smoke tests)')
  ap.add_argument('--force', action='store_true', help='Re-extract even if pickle exists')
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  fail_log_path = os.path.join(args.out_dir, 'failed.log')
  progress_path = os.path.join(args.out_dir, 'progress.jsonl')

  routes = parse_route_list(args.routes)
  if args.limit is not None:
    routes = routes[: args.limit]

  todo: list[tuple[str, str]] = []
  skipped = 0
  for name, ident in routes:
    p = pickle_path_for(name, args.out_dir)
    if os.path.exists(p) and not args.force:
      skipped += 1
      continue
    todo.append((name, ident))

  print(f"routes total: {len(routes)}  skipped (pickle exists): {skipped}  to process: {len(todo)}")
  if not todo:
    return

  stats = {'ok': 0, 'extract_error': 0, 'crash': 0, 'by_category': {}}
  t_start = time.time()

  # Make workers ignore SIGINT so parent can clean up
  def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

  try:
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex, \
         open(fail_log_path, 'a') as fail_log, \
         open(progress_path, 'a') as progress_log:
      futures = {ex.submit(_worker, name, ident, args.out_dir, args.max_seconds): name for name, ident in todo}
      done = 0
      for fut in as_completed(futures):
        rid = futures[fut]
        try:
          info = fut.result()
        except Exception as e:
          info = {'name': rid, 'status': 'crash', 'error': str(e),
                  'traceback': traceback.format_exc(), 'wall_s': 0.0,
                  'category': categorise_failure(str(e))}
        done += 1
        stats[info['status']] = stats.get(info['status'], 0) + 1
        if info['status'] == 'crash':
          cat = info.get('category', 'other')
          stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
          fail_log.write(f"{info['name']}  {cat}  {info['error']!r}\n")
          fail_log.write(info.get('traceback', '') + "\n---\n")
          fail_log.flush()
        progress_log.write(json.dumps({
          'name': info['name'],
          'status': info['status'],
          'error': info.get('error'),
          'category': info.get('category'),
          'wall_s': info.get('wall_s'),
          'learn_samples': info.get('learn_samples', 0),
        }) + '\n')
        progress_log.flush()
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1e-3)
        eta = (len(todo) - done) / max(rate, 1e-3)
        if done % 25 == 0 or done == len(todo):
          ok = stats.get('ok', 0)
          fail = stats.get('crash', 0) + stats.get('extract_error', 0)
          print(f"  [{done}/{len(todo)}]  ok={ok} fail={fail}  cats={stats.get('by_category', {})}  rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)
  except KeyboardInterrupt:
    print("interrupted; saving what we have", file=sys.stderr)
    raise

  ok = stats.get('ok', 0)
  ee = stats.get('extract_error', 0)
  crash = stats.get('crash', 0)
  print(f"\nDONE  ok={ok}  extract_error={ee}  crash={crash}")
  if stats['by_category']:
    print("crash categories:")
    for k, v in sorted(stats['by_category'].items(), key=lambda kv: -kv[1]):
      print(f"  {k}: {v}")
  print(f"failed.log : {fail_log_path}")
  print(f"progress    : {progress_path}")


if __name__ == '__main__':
  main()
