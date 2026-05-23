"""Fleet-sweep orchestrator.

Reads a manifest CSV, runs `extract.run` for each route in a multiprocessing
pool, appends results to a parquet (resumable), and after the sweep produces
per-dongle summaries + decision.md + a basic HTML report.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .aggregate import aggregate_per_dongle, fleet_decision, load_per_route_rows
from .extract import ExtractResult, pack_features, run as extract_run
from .manifest import already_processed, load_manifest, ManifestRow


PER_ROUTE_SCHEMA = pa.schema([
  pa.field("route_key", pa.string()),
  pa.field("dongle_id", pa.string()),
  pa.field("route_id", pa.string()),
  pa.field("branch", pa.string()),
  pa.field("status", pa.string()),
  pa.field("error", pa.string()),
  pa.field("car_fingerprint", pa.string()),
  pa.field("car_vin", pa.string()),
  pa.field("total_seconds", pa.float64()),
  pa.field("engaged_seconds", pa.float64()),
  pa.field("n_samples_total", pa.int64()),
  pa.field("n_samples_gated", pa.int64()),
  pa.field("elapsed_s", pa.float64()),
  pa.field("features_blob", pa.binary()),
])


def _result_to_row(r: ExtractResult) -> dict:
  blob = pack_features(r.features) if r.features else b""
  return {
    "route_key": r.route_key,
    "dongle_id": r.dongle_id,
    "route_id": r.route_id,
    "branch": r.branch,
    "status": r.status,
    "error": r.error,
    "car_fingerprint": r.car_fingerprint,
    "car_vin": r.car_vin,
    "total_seconds": r.total_seconds,
    "engaged_seconds": r.engaged_seconds,
    "n_samples_total": r.n_samples_total,
    "n_samples_gated": r.n_samples_gated,
    "elapsed_s": r.elapsed_s,
    "features_blob": blob,
  }


def _append_parquet(rows: list[dict], path: Path) -> None:
  if not rows:
    return
  table = pa.Table.from_pylist(rows, schema=PER_ROUTE_SCHEMA)
  if path.exists():
    existing = pq.read_table(path, schema=PER_ROUTE_SCHEMA)
    table = pa.concat_tables([existing, table])
  pq.write_table(table, path, compression="zstd")


def _worker(args: tuple[str, str, str, str]) -> ExtractResult:
  dongle_id, route_id, branch, rlog_url = args
  return extract_run(dongle_id, route_id, branch, rlog_url)


def sweep(manifest_rows: list[ManifestRow], out_dir: Path, workers: int,
          per_route_timeout_s: float = 300.0, batch_size: int = 32) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  per_route_path = out_dir / "per_route.parquet"
  failures_path = out_dir / "failures.csv"

  done = already_processed(per_route_path)
  todo = [r for r in manifest_rows if r.route_key not in done]
  print(f"sweep: {len(todo)} routes pending, {len(done)} already processed.", file=sys.stderr)

  fail_f = failures_path.open("a")
  if failures_path.stat().st_size == 0:
    fail_f.write("route_key,status,error\n")

  rows_buffer: list[dict] = []
  t0 = time.monotonic()
  completed = 0

  if workers <= 1:
    for row in todo:
      try:
        r = _worker((row.dongle_id, row.route_id, row.branch, row.rlog_url))
      except Exception as e:
        r = ExtractResult(row.route_key, row.dongle_id, row.route_id, row.branch,
                          status="error", error=str(e))
      rows_buffer.append(_result_to_row(r))
      if r.status != "ok":
        fail_f.write(f"{r.route_key},{r.status},{r.error[:200].replace(chr(10),' ')}\n")
        fail_f.flush()
      completed += 1
      if len(rows_buffer) >= batch_size:
        _append_parquet(rows_buffer, per_route_path); rows_buffer.clear()
        print(f"  [{completed}/{len(todo)}] {time.monotonic()-t0:.0f}s elapsed", file=sys.stderr)
  else:
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as exe:
      futures = {exe.submit(_worker, (r.dongle_id, r.route_id, r.branch, r.rlog_url)): r for r in todo}
      for fut in list(futures.keys()):
        row_in = futures[fut]
        try:
          r = fut.result(timeout=per_route_timeout_s)
        except FutureTimeout:
          r = ExtractResult(row_in.route_key, row_in.dongle_id, row_in.route_id, row_in.branch,
                            status="timeout", error=f"exceeded {per_route_timeout_s:.0f}s")
        except Exception as e:
          r = ExtractResult(row_in.route_key, row_in.dongle_id, row_in.route_id, row_in.branch,
                            status="error", error=str(e))
        rows_buffer.append(_result_to_row(r))
        if r.status != "ok":
          fail_f.write(f"{r.route_key},{r.status},{r.error[:200].replace(chr(10),' ')}\n")
          fail_f.flush()
        completed += 1
        if len(rows_buffer) >= batch_size:
          _append_parquet(rows_buffer, per_route_path); rows_buffer.clear()
          elapsed = time.monotonic() - t0
          rate = completed / max(elapsed, 1e-3)
          eta = (len(todo) - completed) / max(rate, 1e-9)
          print(f"  [{completed}/{len(todo)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", file=sys.stderr)

  _append_parquet(rows_buffer, per_route_path)
  fail_f.close()
  return per_route_path


def summarize(out_dir: Path) -> dict:
  from collections import defaultdict
  from .eps_model import fit_dongle, fit_fleet_then_test_per_dongle, write_eps_model_report

  per_route_path = out_dir / "per_route.parquet"
  rows = load_per_route_rows(str(per_route_path))
  dongles = aggregate_per_dongle(rows)
  decision = fleet_decision(dongles)

  # plant model fit: route-wise train/test per dongle + cross-dongle fleet generalization
  per_dongle_route_features: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    if r.get("status") == "ok" and r.get("features"):
      per_dongle_route_features[r["dongle_id"]].append(r["features"])
  per_dongle_fits = {d: fit_dongle(d, routes) for d, routes in per_dongle_route_features.items()}
  fleet_fit, cross_r2 = fit_fleet_then_test_per_dongle(per_dongle_route_features)
  write_eps_model_report(per_dongle_fits, fleet_fit, cross_r2, out_dir / "eps_model.md")
  decision["eps_model_fit_label"] = fleet_fit.fit_label
  decision["eps_model_cross_dongle_r2"] = cross_r2

  # write per_dongle.parquet (lightweight: drop pooled_features blob to keep file small)
  dongle_rows = []
  for d in dongles:
    dongle_rows.append({
      "dongle_id": d.dongle_id,
      "n_routes": d.n_routes,
      "total_seconds": d.total_seconds,
      "engaged_seconds": d.engaged_seconds,
      "gated_samples": d.gated_samples,
      "scalar_tracking_score": d.scalar_tracking_score,
      "car_fingerprint": d.fingerprint,
      "car_vin": d.vin,
      "winner_hypothesis": d.hypothesis["winner"],
      "fits_json": json.dumps({
        n: {"params": f["params"], "aic": _finite(f["aic"]), "failed": f["failed"], "reason": f["reason"], "n": f["n"]}
        for n, f in d.hypothesis["fits"].items()
      }),
    })
  if dongle_rows:
    pq.write_table(pa.Table.from_pylist(dongle_rows), out_dir / "per_dongle.parquet", compression="zstd")

  (out_dir / "decision.md").write_text(_render_decision(decision, dongles))
  try:
    from .report import write_html_report
    write_html_report(dongles, decision, out_dir / "report.html")
  except Exception as e:
    print(f"warning: HTML report failed ({e}); decision.md still written", file=sys.stderr)
  return decision


def _finite(x):
  return None if (x is None or not np.isfinite(x)) else float(x)


def _render_decision(decision: dict, dongles: list) -> str:
  lines = ["# ID4 MK1 lateral-tracking fleet decision\n"]
  lines.append(f"- dongles total: {decision['n_dongles_total']}")
  lines.append(f"- dongles qualified (>=200 gated samples): {decision['n_dongles_qualified']}")
  lines.append(f"- threshold for fleet-level conclusion: >= {decision['min_dongles_for_fleet_conclusion']} dongles")
  lines.append(f"- fleet conclusion valid: {decision['fleet_conclusion_valid']}")
  if decision["warnings"]:
    lines.append("\n## Warnings")
    for w in decision["warnings"]:
      lines.append(f"- {w}")
  if decision.get("score_percentiles"):
    lines.append("\n## Cross-dongle tracking-score distribution (RMS residual, lower=better)")
    for p, v in decision["score_percentiles"].items():
      lines.append(f"- p{p}: {v:.3e} rad/m")
    lines.append(f"- IQR/median ratio: {decision.get('score_iqr_over_median_ratio', float('nan')):.2f}")
  lines.append("\n## Winning hypothesis per dongle")
  for h, n in sorted(decision["per_hypothesis_count"].items(), key=lambda kv: -kv[1]):
    lines.append(f"- {h}: {n}")
  lines.append(f"\n## Recommendation: **{decision['recommendation']}**\n")
  if "rationale" in decision:
    lines.append(decision["rationale"])
  if dongles:
    lines.append("\n## Per-dongle leaderboard (lowest tracking-score first)")
    lines.append("| dongle | routes | engaged_s | gated_n | score (RMS rad/m) | winner | VIN |")
    lines.append("|--------|--------|-----------|---------|-------------------|--------|-----|")
    for d in dongles[:50]:
      lines.append(
        f"| {d.dongle_id} | {d.n_routes} | {d.engaged_seconds:.0f} | "
        f"{d.gated_samples} | {d.scalar_tracking_score:.3e} | {d.hypothesis['winner']} | {d.vin} |"
      )
  return "\n".join(lines) + "\n"


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--manifest", required=True, type=Path, help="CSV with columns: dongle_id,route_id[,branch]")
  ap.add_argument("--out", required=True, type=Path, help="output directory")
  ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
  ap.add_argument("--timeout", type=float, default=300.0, help="per-route timeout in seconds")
  ap.add_argument("--summarize-only", action="store_true", help="skip extraction, only re-summarize existing parquet")
  args = ap.parse_args(argv)

  args.out.mkdir(parents=True, exist_ok=True)
  if not args.summarize_only:
    rows = load_manifest(args.manifest)
    if not rows:
      print("empty manifest", file=sys.stderr); return 2
    sweep(rows, args.out, workers=args.workers, per_route_timeout_s=args.timeout)
  decision = summarize(args.out)
  print(json.dumps({"recommendation": decision["recommendation"],
                    "n_dongles_qualified": decision["n_dongles_qualified"]}, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
