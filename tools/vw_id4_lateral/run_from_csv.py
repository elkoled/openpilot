#!/usr/bin/env python3
"""
CSV-driven fleet driver — reads eps_seglist.csv (segment, dongle, platform,
mean_v, max_v, engaged_frac, rlog_url) and runs extract() per row with a
multiprocessing pool. Resumable.

Output:
  per_segment.jsonl — one record per processed segment
  cache/<dongle>/<safe_route>.npz — engaged 100 Hz timeline per segment
                                    (consumed by plant_fit.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

from openpilot.tools.vw_id4_lateral.extract_segment import extract, record_to_dict


def load_csv(path: str) -> list[dict]:
  rows = []
  with open(path) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
      rows.append(row)
  return rows


def already_processed(out_path: str) -> set[str]:
  done: set[str] = set()
  if not os.path.exists(out_path):
    return done
  with open(out_path) as f:
    for line in f:
      try:
        r = json.loads(line)
        done.add(r.get("route", ""))
      except Exception:
        continue
  return done


def _worker(payload: dict) -> dict:
  try:
    rec = extract(
      route=payload["segment"],
      dongle=payload["dongle"],
      fingerprint=payload.get("platform", "VOLKSWAGEN_ID4_MK1"),
      rlog_url=payload["rlog_url"],
      timeline_cache_dir=payload.get("timeline_cache_dir", ""),
    )
    return record_to_dict(rec)
  except Exception as e:
    return {
      "route": payload.get("segment", ""),
      "dongle": payload.get("dongle", ""),
      "ok": False,
      "reason": f"worker_exception:{type(e).__name__}:{str(e)[:200]}",
    }


def run(csv_path: str, out_path: str, workers: int, limit: int | None,
        dongle_filter: str, cache_dir: str, balance_dongles: bool) -> None:
  rows = load_csv(csv_path)
  if dongle_filter:
    keep = set(dongle_filter.split(","))
    rows = [r for r in rows if r["dongle"] in keep]
  done = already_processed(out_path)
  todo_rows = [r for r in rows if r["segment"] not in done]

  if balance_dongles:
    # Interleave by dongle so a small dongle isn't starved at the tail.
    by_d: dict[str, list[dict]] = {}
    for r in todo_rows:
      by_d.setdefault(r["dongle"], []).append(r)
    interleaved = []
    cursors = {d: 0 for d in by_d}
    while any(cursors[d] < len(by_d[d]) for d in by_d):
      for d in sorted(by_d):
        if cursors[d] < len(by_d[d]):
          interleaved.append(by_d[d][cursors[d]])
          cursors[d] += 1
    todo_rows = interleaved

  if limit is not None:
    todo_rows = todo_rows[:limit]

  print(f"[csv] csv_rows={len(rows)} todo={len(todo_rows)} done={len(done)} "
        f"workers={workers} cache_dir={cache_dir!r}", flush=True)

  payloads = [
    {
      "segment": r["segment"],
      "dongle": r["dongle"],
      "platform": r["platform"],
      "rlog_url": r["rlog_url"],
      "timeline_cache_dir": cache_dir,
    }
    for r in todo_rows
  ]

  t0 = time.time()
  with open(out_path, "a") as out:
    with Pool(workers) as pool:
      for i, rec_dict in enumerate(pool.imap_unordered(_worker, payloads, chunksize=1)):
        out.write(json.dumps(rec_dict) + "\n")
        out.flush()
        if (i + 1) % 20 == 0 or i + 1 == len(payloads):
          dt = time.time() - t0
          rate = (i + 1) / max(dt, 1e-6)
          eta = (len(payloads) - (i + 1)) / max(rate, 1e-6)
          print(f"[csv] {i+1}/{len(payloads)}  rate={rate:.2f}/s  eta={eta/60:.1f}min  "
                f"last={rec_dict.get('route','?')[:60]} ok={rec_dict.get('ok')}", flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--csv", default=os.path.expanduser("~/eps_seglist.csv"))
  p.add_argument("--out", default="per_segment.jsonl")
  p.add_argument("--cache-dir", default="cache",
                 help="Directory to write per-segment timeline npz sidecars")
  p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
  p.add_argument("--limit", type=int, default=None)
  p.add_argument("--dongle", default="", help="Comma-separated dongle ID filter")
  p.add_argument("--balance-dongles", action="store_true",
                 help="Interleave todo rows by dongle (useful for partial runs)")
  args = p.parse_args()
  if args.cache_dir:
    os.makedirs(args.cache_dir, exist_ok=True)
  run(args.csv, args.out, args.workers, args.limit, args.dongle, args.cache_dir,
      args.balance_dongles)


if __name__ == "__main__":
  sys.exit(main())
