#!/usr/bin/env python3
"""
Fleet driver — reads manifest.jsonl, dispatches accessible routes to a worker
pool, appends one JSONL record per route to per_segment.jsonl. Resumable: any
route whose record is already present in the output is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

from openpilot.tools.vw_id4_lateral.extract_segment import extract, record_to_dict


def already_processed(out_path: str) -> set[str]:
  done: set[str] = set()
  if not os.path.exists(out_path):
    return done
  with open(out_path) as f:
    for line in f:
      try:
        r = json.loads(line)
        done.add(r["route"])
      except Exception:
        continue
  return done


def _worker(payload: dict) -> dict:
  try:
    rec = extract(**payload)
    return record_to_dict(rec)
  except Exception as e:
    return {
      "route": payload["route"],
      "dongle": payload.get("dongle", ""),
      "ok": False,
      "reason": f"worker_exception:{type(e).__name__}:{str(e)[:200]}",
    }


def load_manifest(path: str) -> list[dict]:
  rows: list[dict] = []
  with open(path) as f:
    for line in f:
      try:
        rows.append(json.loads(line))
      except Exception:
        continue
  return rows


def run(manifest_path: str, out_path: str, workers: int, limit: int | None) -> None:
  manifest = load_manifest(manifest_path)
  accessible = [r for r in manifest if r.get("accessible")]
  done = already_processed(out_path)
  todo = [r for r in accessible if r["route"] not in done]
  if limit is not None:
    todo = todo[:limit]
  print(f"[fleet] manifest={len(manifest)} accessible={len(accessible)} "
        f"already_done={len(done)} todo={len(todo)} workers={workers}", flush=True)

  payloads = [
    {
      "route": r["route"],
      "dongle": r.get("dongle", ""),
      "fingerprint": r.get("fingerprint", ""),
      "vin": r.get("vin", ""),
      "build_year": r.get("build_year", ""),
      "steer_control_type": r.get("steer_control_type", ""),
      "has_live_curvature_parameters": r.get("has_live_curvature_parameters", False),
    }
    for r in todo
  ]

  t0 = time.time()
  with open(out_path, "a") as out:
    with Pool(workers) as pool:
      for i, rec_dict in enumerate(pool.imap_unordered(_worker, payloads, chunksize=1)):
        out.write(json.dumps(rec_dict) + "\n")
        out.flush()
        if (i + 1) % 10 == 0:
          dt = time.time() - t0
          rate = (i + 1) / max(dt, 1e-6)
          eta = (len(payloads) - (i + 1)) / max(rate, 1e-6)
          print(f"[fleet] {i+1}/{len(payloads)}  rate={rate:.2f}/s  eta={eta/60:.1f}min  "
                f"last={rec_dict.get('route','?')} ok={rec_dict.get('ok')}", flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--manifest", required=True)
  p.add_argument("--out", default="per_segment.jsonl")
  p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
  p.add_argument("--limit", type=int, default=None, help="Process only the first N todo routes")
  args = p.parse_args()
  run(args.manifest, args.out, args.workers, args.limit)


if __name__ == "__main__":
  sys.exit(main())
