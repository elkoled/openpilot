#!/usr/bin/env python3
"""
Driver for fleet_lateral_extract over a list of routes.

Input formats accepted (auto-detected):
  - whitespace-separated:  <dongle_id> <route_id>
  - slash-separated:       <dongle_id>/<route_id>[/<a|q|r>]
  - lines starting with '#' are ignored

Each route is processed in a worker process with a wall-clock timeout.
Per-route NPZ is written to  <out_dir>/<dongle_id>/<route_id>.npz.
A manifest TSV is written to <out_dir>/manifest.tsv with one row per route.

Usage:
  python fleet_lateral_run.py <route_list>  [<out_dir>]  [--workers N] [--timeout S]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _parse_route_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.replace("\t", " ").split()
    if len(parts) == 1 and "/" in parts[0]:
        a, b = parts[0].split("/", 1)
        b = b.split("/", 1)[0]   # strip optional /a /q /r suffix
        return a, b
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _worker(args):
    """Run inside a child process.  Returns a manifest row dict.

    args = (dongle_id, route_id, out_dir, source_path) where source_path is
    either an openpilot route spec ('<dongle>/<route>/a') or a direct rlog URL.
    """
    dongle_id, route_id, out_dir, source_path = args
    out_path = Path(out_dir) / dongle_id / f"{route_id}.npz"
    if out_path.exists():
        # Quick read of header
        try:
            import numpy as np
            with np.load(out_path, allow_pickle=True) as data:
                h = json.loads(str(data["header_json"]))
            return {
                "dongle_id": dongle_id, "route_id": route_id,
                "status": "cached", "accepted": int(np.load(out_path)["n"].sum()),
                "duration_s": h.get("duration_s", 0.0),
                "engaged_s": h.get("engaged_s", 0.0),
                "fp": h.get("car_fingerprint", ""),
                "branch": h.get("git_branch", ""),
                "qfk_bus": h.get("qfk_bus", -1),
                "elapsed_s": 0.0,
                "note": "from-cache",
            }
        except Exception:  # noqa: BLE001
            out_path.unlink(missing_ok=True)

    t0 = time.time()
    try:
        from openpilot.tools.lateral_maneuvers.fleet_lateral_extract import extract_route
        res = extract_route(source_path, out_path,
                            dongle_id=dongle_id, route_id=route_id)
        import numpy as np  # noqa: F401 (also imported by extract_route)
        h = json.loads(res["header_json"])
        accepted = int(res["n"].sum())
        return {
            "dongle_id": dongle_id, "route_id": route_id,
            "status": "ok" if accepted > 0 else "empty",
            "accepted": accepted,
            "duration_s": h.get("duration_s", 0.0),
            "engaged_s": h.get("engaged_s", 0.0),
            "fp": h.get("car_fingerprint", ""),
            "branch": h.get("git_branch", ""),
            "qfk_bus": h.get("qfk_bus", -1),
            "elapsed_s": time.time() - t0,
            "note": ";".join(h.get("notes", []))[:200],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "dongle_id": dongle_id, "route_id": route_id,
            "status": f"error:{type(e).__name__}",
            "accepted": 0, "duration_s": 0.0, "engaged_s": 0.0,
            "fp": "", "branch": "", "qfk_bus": -1,
            "elapsed_s": time.time() - t0,
            "note": (str(e) + " | " + traceback.format_exc().splitlines()[-1])[:200],
        }


def _init_worker():
    # SIGINT in worker -> propagate to parent, don't fight it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("route_list", help="Text file with routes (one '<dongle> <route>' "
                                       "or '<dongle>/<route>' per line), OR a CSV with "
                                       "columns 'segment' and 'rlog_url' (use --csv).")
    p.add_argument("out_dir", nargs="?", default="/home/batman/curv_analysis/fleet_summaries",
                   help="Output directory for per-route NPZ + manifest.tsv")
    p.add_argument("--csv", action="store_true",
                   help="Treat route_list as a CSV with 'segment' (=<dongle>/<route>--<seg>) "
                        "and 'rlog_url' columns; pass URLs directly to LogReader.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-route timeout in seconds (default 300s)")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process first N routes (debug / smoke test)")
    args = p.parse_args(argv[1:])

    # routes will be list of (dongle, route_id, source_path) tuples.
    routes: list[tuple[str, str, str]] = []
    if args.csv:
        import csv as _csv
        with open(args.route_list) as f:
            for row in _csv.DictReader(f):
                seg = row["segment"]
                # seg = "<dongle>/<route>--<seg_num>"
                dongle = row.get("dongle") or seg.split("/", 1)[0]
                route_with_seg = seg.split("/", 1)[1] if "/" in seg else seg
                url = row["rlog_url"]
                routes.append((dongle, route_with_seg, url))
    else:
        with open(args.route_list) as f:
            for ln in f:
                r = _parse_route_line(ln)
                if r is not None:
                    routes.append((r[0], r[1], f"{r[0]}/{r[1]}/a"))
    if args.limit > 0:
        routes = routes[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.tsv"

    print(f"routes={len(routes)} workers={args.workers} timeout={args.timeout}s out={out_dir}",
          flush=True)

    task_args = [(d, r, str(out_dir), src) for (d, r, src) in routes]
    rows: list[dict] = []
    t_start = time.time()

    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                       maxtasksperchild=4) as pool:
        async_results = [pool.apply_async(_worker, (a,)) for a in task_args]
        for i, ar in enumerate(async_results):
            try:
                row = ar.get(timeout=args.timeout)
            except mp.TimeoutError:
                d, r, _, _ = task_args[i]
                row = {"dongle_id": d, "route_id": r,
                       "status": "timeout", "accepted": 0,
                       "duration_s": 0.0, "engaged_s": 0.0, "fp": "", "branch": "",
                       "qfk_bus": -1, "elapsed_s": args.timeout,
                       "note": f"timeout after {args.timeout}s"}
            except Exception as e:  # noqa: BLE001
                d, r, _, _ = task_args[i]
                row = {"dongle_id": d, "route_id": r,
                       "status": f"poolerror:{type(e).__name__}",
                       "accepted": 0, "duration_s": 0.0, "engaged_s": 0.0,
                       "fp": "", "branch": "", "qfk_bus": -1,
                       "elapsed_s": 0.0, "note": str(e)[:200]}
            rows.append(row)
            elapsed_total = time.time() - t_start
            print(f"[{i+1}/{len(task_args)}] {row['dongle_id']}/{row['route_id']} "
                  f"status={row['status']} acc={row['accepted']} "
                  f"engaged={row['engaged_s']:.0f}s fp={row['fp']} "
                  f"({row['elapsed_s']:.1f}s; total={elapsed_total:.0f}s)",
                  flush=True)

    fieldnames = ["dongle_id", "route_id", "status", "accepted",
                  "duration_s", "engaged_s", "fp", "branch", "qfk_bus",
                  "elapsed_s", "note"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Summary
    n = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_empty = sum(1 for r in rows if r["status"] == "empty")
    n_cached = sum(1 for r in rows if r["status"] == "cached")
    n_err = sum(1 for r in rows if r["status"].startswith(("error", "timeout", "poolerror")))
    total_engaged = sum(r["engaged_s"] for r in rows)
    total_acc = sum(r["accepted"] for r in rows)
    print(f"\nDONE  total={n}  ok={n_ok}  empty={n_empty}  cached={n_cached}  err={n_err}")
    print(f"  total_engaged={total_engaged:.0f}s   total_accepted_samples={total_acc}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
