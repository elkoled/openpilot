#!/bin/bash
# End-to-end EPS-model pipeline.
# Usage:  ./run_eps_pipeline.sh <route_or_csv> <out_dir> [--csv]
set -euo pipefail

SRC="${1:?usage: run_eps_pipeline.sh <route_list_or_csv> <out_dir> [--csv]}"
OUT="${2:?missing out_dir}"
shift 2

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

echo "=== 1/4 extract ==="
python3 "$HERE/fleet_lateral_run.py" "$SRC" "$OUT" "$@" --workers 16 --timeout 240

echo
echo "=== 2/4 aggregate ==="
python3 "$HERE/fleet_lateral_aggregate.py" "$OUT" --min_dongle_engaged_s 300

echo
echo "=== 3/4 fit & validate ==="
python3 "$HERE/fleet_lateral_model.py" "$OUT/per_dongle_bucket.npz" --out_dir "$OUT"

echo
echo "=== 4/4 patch decision ==="
python3 "$HERE/fleet_lateral_patch.py" "$OUT/model.json" --out_dir "$OUT" || true

echo
echo "=== artifacts ==="
ls -la "$OUT"
